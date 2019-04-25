from util_tf import tf, placeholder, dense, pos_enc, mh_attention, mlp, padding_mask, causal_mask
from util_np import np


def transformer(data, batch_size, emb_dim, heads, layers, att_dim, mlp_dim,
                dim_vocab, max_len, lbl_smooth, eos, dropout, train, warmup):


    def decoder(inpt, att_dim, emb_dim, mlp_dim, heads, enc_output, dropout, train, mask_causal=None, mask_pad=None):
        #Multihead self attention
        with tf.variable_scope("dec_att"):
            z = mh_attention(inpt, att_dim, emb_dim, heads, dropout, train, mask=mask_causal)

        # encoder decoder multihead attention
        with tf.variable_scope("dec_enc_att"):
            z = mh_attention(z, att_dim, emb_dim, heads, dropout, train, memory=enc_output, mask=mask_pad)

        with tf.variable_scope("dec_mlp"):
            #feedforward
            z = mlp(z, mlp_dim, dropout, train)

        return z

    def encoder(inpt, att_dim, emb_dim, mlp_dim, heads, dropout, train, mask=None):
        # Multihead self attention
        with tf.variable_scope("enc_att"):
            z = mh_attention(inpt, att_dim, emb_dim, heads, dropout, train, mask=mask)
        #feedforward
        with tf.variable_scope("enc_mlp"):
            z = mlp(z, mlp_dim, dropout, train)

        return z


    #########
    # INPUT #
    #########
    train = placeholder(tf.bool, (), train, 'training')
    with tf.variable_scope("Input"):
        with tf.variable_scope("enc_input"):
            enc_inpt = placeholder(tf.int32, (None, None), data[0], 'enc_inpt')

        with tf.variable_scope("dec_data"):
            dec_data = placeholder(tf.int32, (None, None), data[1], 'dec_inpt')

        with tf.variable_scope("dec_inpt"):
            dec_inpt = dec_data[:,:-1]

        with tf.variable_scope("tgt"):
            tgt = dec_data[:, 1:]
            oh_tgt = tf.one_hot(tgt, dim_vocab, name="oh_tgt")


    ###########
    # MASKING #
    ###########
    with tf.variable_scope("mask_pad"):
        mask_pad = padding_mask(enc_inpt, eos)
    with tf.variable_scope("mask_causal"):
        mask_causal = causal_mask(tf.shape(dec_inpt)[-1])
    #with tf.variable_scope("mask_logits"):
    #    mask_logits = tf.not_equal(tgt, eos)


    #############
    # EMBEDDING #
    #############
    with tf.variable_scope("enc_embed"):
        # (b,t) -> (b,t,emb_dim)
        enc_emb = tf.get_variable('enc_emb', (dim_vocab, emb_dim))
        emb_enc = tf.gather(enc_emb * (emb_dim ** 0.5), enc_inpt)
    # decoder
    with tf.variable_scope("dec_emb"):
        # (b,t) -> (b,t,emb_dim)
        dec_emb = tf.get_variable('dec_emb', (dim_vocab, emb_dim))
        emb_dec = tf.gather(dec_emb * (emb_dim ** 0.5), dec_inpt)


    #######################
    # POSITIONAL ENCODING #
    #######################
    with tf.variable_scope("sinusoid"):
        sinusoid = tf.constant(pos_enc(max_len, emb_dim))
    # encoder
    with tf.variable_scope("pos_enc_enc"):
        z = emb_enc + sinusoid[:tf.shape(emb_enc)[1]]
        enc_output = tf.layers.dropout(z, rate=dropout, training=train)
    # decoder
    with tf.variable_scope("pos_enc_dec"):
        z = emb_dec + sinusoid[:tf.shape(emb_dec)[1]]
        z = tf.layers.dropout(z, rate=dropout, training=train)


    ###########
    # ENCODER #
    ###########
    with tf.variable_scope("encoder"):
        for layer in range(layers):
            with tf.variable_scope("enc_layer_{}".format(layer+1)):
                enc_output = encoder(enc_output, att_dim, emb_dim, mlp_dim, heads,
                                     dropout, train, mask=mask_pad)


    ###########
    # DECODER #
    ###########
    with tf.variable_scope("decoder"):
        for layer in range(layers):
            with tf.variable_scope("dec_layer_{}".format(layer+1)):
                z = decoder(z, att_dim, emb_dim, mlp_dim, heads, enc_output, dropout,
                            train, mask_causal=mask_causal, mask_pad=mask_pad)


    ##########
    # OUTPUT #
    ##########
    with tf.variable_scope("learn_rate"):
        step = tf.train.get_or_create_global_step()
        t = tf.to_float(step + 1)
        learn_rate = (emb_dim ** -0.5) * tf.minimum(t ** -0.5, t * (warmup ** -1.5))

    with tf.variable_scope('logits'):
        #z = tf.boolean_mask(z, mask_logits)
        z_shp = tf.shape(z) # b,t,d
        z = tf.reshape(z, [z_shp[0] * z_shp[1], mlp_dim])  # b,t,d' -> b*t,d
        # dec_emb * (emb_dim ** 0.5) ???
        z = tf.matmul(z, dec_emb, transpose_b=True)  # b*t,d @ d,d' -> b*t,d'
        logits = tf.reshape(z, [z_shp[0], z_shp[1], dim_vocab])

    with tf.variable_scope('xntropy'):
        xntropy = tf.losses.softmax_cross_entropy(tf.reshape(oh_tgt, (-1, dim_vocab)),
                                                  logits=z,
                                                  label_smoothing=lbl_smooth,
                                                  reduction=tf.losses.Reduction.NONE)

    with tf.variable_scope('loss'):
        loss = tf.reduce_mean(xntropy)

    with tf.variable_scope('pred'):
        pred = tf.argmax(logits, -1, output_type=tf.int32)

    with tf.variable_scope('acc'):
        acc = tf.reduce_mean(tf.to_float(tf.equal(tgt, pred)))

    train_step = tf.train.AdamOptimizer(learning_rate=learn_rate,
                                        beta1=0.9,
                                        beta2=0.98,
                                        epsilon=1e-09).minimize(loss, step)

    return dict(enc_inpt=enc_inpt,
                enc_output=enc_output,
                mask_pad=mask_pad,
                dec_data=dec_data,
                dec_inpt=dec_inpt,
                train=train,
                pred=pred,
                acc=acc,
                step=step,
                loss=loss,
                train_step=train_step)

# MATRIX MULTIPLICATION
#   ab @   bc  ->   ac
#  abc @  ace  ->  abe
# abcd @ abde  -> abce

# BROADCASTING
### dimension 1
# |7 8 9| ==> |7 8 9|
#             |7 8 9|
#
# (2,3)+(3)  if batch: (b,2,3) ->
#
# |1 2 3| + |7 8 9|  = | 8  9 10|
# |4 5 6|              |11 12 13|
#
#
### dimension 0
# |7| ==> |7 7 7|
# |8|     |8 8 8|
# |9|     |9 9 9|
#
# (2,3) + (1,3)
