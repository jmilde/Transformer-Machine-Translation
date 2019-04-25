from model import transformer
from util_tf import tf, pipe, batch
from util_io import pform
from util_np import np, vpack,  partition
from hyperparameters import parameters as params
import util_sp as sp
from tqdm import tqdm
import os

def train(trial, data_path, path_ckpt, path_log, gpu, max_len, batch_size, batch_size_valid,
          dropout, emb_dim, heads, layers, att_dim, mlp_dim, warmup, lbl_smooth, eos, bos, epochs):

    ############# PICK GPU ###################
    if gpu != None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    # form paths
    path_ger_vocab = pform(data_path, "vocab_ger.model")
    path_eng_vocab = pform(data_path, "vocab_eng.model")
    path_traintxt = pform(data_path, "train_txt.npy")
    path_traintgt = pform(data_path, "train_tgt.npy")
    path_validtxt = pform(data_path, "valid_txt.npy")
    path_validtgt = pform(data_path, "valid_tgt.npy")


    # sentence piece model
    vocab_enc = sp.load_spm(path_ger_vocab)
    vocab_enc.SetEncodeExtraOptions("eos") # enable end of sentence symbol
    vocab_dec = sp.load_spm(path_eng_vocab)
    vocab_dec.SetEncodeExtraOptions("bos:eos")# enable beginning/end of sentence symbol
    dim_vocab = vocab_enc.GetPieceSize()


    #validation data
    valid_txt = np.load(path_validtxt)
    valid_tgt = np.load(path_validtgt)

    ##############################################################################################

    def summ(step):
        fetches = model['acc'], model['loss']
        results = map(np.mean, zip(*(
            sess.run(fetches,
                     {model['enc_inpt']: vpack(valid_txt[i:j], (batch_size_valid, max(map(len, valid_txt[i:j]))), eos, np.int32),
                      model['dec_data']: vpack(valid_tgt[i:j], (batch_size_valid, max(map(len, valid_tgt[i:j]))), eos, np.int32),
                      model['train']: False})
            for i, j in partition(len(valid_txt), batch_size_valid, discard=False))))
        results = list(results)
        wrtr.add_summary(sess.run(summary_test, dict(zip(fetches, results))), step)
        wrtr.add_summary(sess.run(summary_train, {model['train']: False}), step)
        wrtr.flush()
        return results


    ##############################################################################################

    # INPUT
    batch_fn = lambda: batch(batch_size, path_traintxt, path_traintgt, eos, 25)
    data = pipe(batch_fn, (tf.int32, tf.int32), prefetch=4)
    model = transformer(data, batch_size, emb_dim, heads, layers, att_dim, mlp_dim, dim_vocab, \
                        max_len, lbl_smooth, eos, dropout, True, warmup)

    sess = tf.InteractiveSession()
    saver = tf.train.Saver()

    ### to test teacher forcing
    #pretrain = "transformer"
    #saver.restore(sess, pform(path_ckpt, pretrain))
    #x = sess.run(model['pred'])
    #vocab_dec.decode_ids(x[3].tolist())

    tf.global_variables_initializer().run()


    summary_train = tf.summary.merge((tf.summary.scalar('train_acc', model['acc']),
                                      tf.summary.scalar('train_loss', model['loss'])))
    summary_test = tf.summary.merge((tf.summary.scalar('test_acc', model['acc']),
                                     tf.summary.scalar('test_loss', model['loss'])))

    wrtr = tf.summary.FileWriter(pform(path_log, trial))
    wrtr.add_graph(sess.graph)

    ### to check model performance
    from util_tf import profile
    profile(sess, wrtr, model['loss'], feed_dict= None, prerun=3, tag='flow')
    wrtr.flush()

    for r in range(epochs):
        for _ in range(28): # = 1 epochs
            for _ in tqdm(range(250), ncols=70):
                sess.run(model['train_step'])

            step = sess.run(model['step'])

            results = summ(step)
            print("valid_acc: {:.3f}, valid_loss: {:.3f}".format(results[0], results[1]))
        saver.save(sess, pform(path_ckpt, trial, r), write_meta_graph=False)


if __name__ == '__main__':
    for key, val in params.items():
        exec(key+"=val")

    train(trial, data_path, path_ckpt, path_log, gpu, max_len, batch_size, batch_size_valid,
          dropout, emb_dim, heads, layers, att_dim, mlp_dim, warmup, lbl_smooth, eos, bos, epochs)
