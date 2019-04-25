from copy import copy
from util import Record
import tensorflow as tf
from util_np import sample, np, vpack


def profile(sess, wrtr, run, feed_dict= None, prerun= 5, tag= 'flow'):
    for _ in range(prerun): sess.run(run, feed_dict)
    meta = tf.RunMetadata()
    sess.run(run, feed_dict, tf.RunOptions(trace_level= tf.RunOptions.FULL_TRACE), meta)
    wrtr.add_run_metadata(meta, tag)


def pipe(*args, prefetch=1, repeat=-1, name='pipe', **kwargs):
    """see `tf.data.Dataset.from_generator`."""
    with tf.variable_scope(name):
        return tf.data.Dataset.from_generator(*args, **kwargs) \
                              .repeat(repeat) \
                              .prefetch(prefetch) \
                              .make_one_shot_iterator() \
                              .get_next()


def placeholder(dtype, shape, x= None, name= None):
    """returns a placeholder with `dtype` and `shape`.

    if tensor `x` is given, converts and uses it as default.

    """
    if x is None: return tf.placeholder(dtype, shape, name)
    try:
        x = tf.convert_to_tensor(x, dtype)
    except ValueError:
        x = tf.cast(x, dtype)
    return tf.placeholder_with_default(x, shape, name)


def batch(size, path_txt, path_tgt, eos, seed):
    """batch function to use with pipe, takes to numpy labels as input"""
    text = np.load(path_txt)
    tgt = np.load(path_tgt)
    b_enc, b_dec = [], []
    for i in sample(len(text), seed):
        if size == len(b_enc):
            b_enc = vpack(b_enc, (size, max(map(len, b_enc))), eos, np.int32)
            b_dec = vpack(b_dec, (size, max(map(len, b_dec))), eos, np.int32)
            yield b_enc, b_dec
            b_enc, b_dec = [], []
        b_enc.append(text[i])
        b_dec.append(tgt[i])



def dense(inpt, dim, name, activation=None):
    #b,t,d -> b,t,d'
    with tf.variable_scope("dense-{}".format(name)):
        in_dim = inpt.get_shape().as_list()[-1] #int(d)
        in_shp = tf.shape(inpt) # b,t,d
        inpt = tf.reshape(inpt, [in_shp[0] * in_shp[1], in_dim])  # b,t,d' -> b*t,d
        w = tf.get_variable("weight", shape=(in_dim, dim))  # d,d'
        d = tf.matmul(inpt, w)  # b*t,d @ d,d' -> b*t,d'
        d = tf.reshape(d, [in_shp[0], in_shp[1], dim])  # b*t,d' -> b,t,d'
        if activation is not None:
            d = activation(d)
        return d

def pos_enc(seq_len, mod_dim, use_np=True):

    if type(seq_len) == int:
        # positions, ascending from 1 to seq_len
        pos = np.arange(1, seq_len+1, dtype=np.float32).reshape(-1, 1)
        # divisor term
        div = (10000.0 ** ((2/mod_dim) * np.arange(mod_dim//2, dtype=np.float32)))
        # calculate sin/cos values
        x = (pos*(1/div)).reshape(-1, 1)  # broadcasting
        # sin = PE(pos,2i), cos = PE(pos,2i+1) (i = model dimension)
        pos_enc = np.concatenate((np.sin(x), np.cos(x)), -1).reshape(seq_len, mod_dim)

    else:
        # tensor with positions, ascending from 1 to seq_len
        pos = tf.cast(tf.expand_dims(tf.range(1, seq_len+1), -1), dtype=tf.float32)
        # divisor term
        div = tf.expand_dims(tf.pow(10000.0, tf.cast(2*tf.range(mod_dim//2)/mod_dim, dtype=tf.float32)),0)
        # calculate sin/cos values
        x = tf.multiply(1/div, pos)  # broadcasting
        # sin = PE(pos,2i), cos = PE(pos,2i+1) (i = model dimension)
        pos_enc = tf.concat((tf.sin(x), tf.cos(x)), -1)

    return pos_enc


def mh_attention(query, att_dim, emb_dim, heads, dropout, train, memory=None, mask=None):
    # query:  (b,s,d)
    # memory: (b,t,d)

    if memory is None:  # self attention
        memory = query

    #   q: (h,b,s,ad)
    # k,v: (h,b,t,ad)
    q = tf.stack(tf.split(dense(query, att_dim*heads, "q"), heads, axis=-1, name="query"))
    k = tf.stack(tf.split(dense(memory, att_dim*heads, "k"), heads, axis=-1, name="key"))
    v = tf.stack(tf.split(dense(memory, att_dim*heads, "v"), heads, axis=-1, name="value"))

    # -> (h,b,s,ad)
    z = attention(q, k, v, att_dim, mask)

    # z = tf.squeeze(tf.concat(tf.split(z, heads, axis=0), axis=-1), axis=0) # -> (b,s,ad*h)
    z = tf.concat(tf.unstack(z), axis= -1)
    z = dense(z, emb_dim, "emb_dim")  # -> (b,s,d)
    z = tf.layers.dropout(z, rate=dropout, training=train)

    # Residual connection + Normalization
    z = tf.contrib.layers.layer_norm(z + query, begin_norm_axis=2)
    return z


def attention(q, k, v, att_dim, mask=None):
    #    q: (b,s,d), k,v: (b,t,d)
    #  out: (b,s,d)

    att_dim = tf.cast(att_dim, dtype=tf.float32)
    out = tf.matmul(q, k, transpose_b=True)*(tf.rsqrt(att_dim)) # (b,s,t)
    if mask is not None:
        out = tf.multiply(out, mask) + (1.0 - mask) * (-1e10) # todo: test -float(inf)
    return tf.matmul(tf.nn.softmax(out), v)


def mlp(inpt, dim, dropout, train, act=tf.nn.relu):
    z = dense(inpt, dim*4, "1", activation=act)
    z = dense(z, dim, "2")
    z = tf.layers.dropout(z, rate=dropout, training=train)
    # Residual connection + Normalization
    z = tf.contrib.layers.layer_norm(z + inpt, begin_norm_axis=2)
    return z


def padding_mask(inpt, pad):
    # (b, t) -> (b,1,t)
    mask = tf.cast(tf.not_equal(inpt, pad), tf.float32)
    mask = tf.expand_dims(mask, 1)
    return mask


def causal_mask(seq_len):
    #tril_mask = tf.expand_dims(tf.contrib.distributions.fill_triangular(tf.ones(n*(n+1)//2)), 0)
    #tril_mask = tf.contrib.distributions.fill_triangular(tf.ones(seq_len*(seq_len+1)//2))
    tril_mask = tf.linalg.LinearOperatorLowerTriangular(tf.ones((seq_len, seq_len))).to_dense()
    return tril_mask #tf.multiply(pad_mask, tril_mask)
