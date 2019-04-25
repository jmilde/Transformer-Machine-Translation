from model import transformer
from util_tf import tf
from util_io import pform, save_txt
from util_np import np, vpack,  partition
import util_sp as sp
from tqdm import tqdm
from os.path import expanduser
import os
from hyperparameters import parameters as params

def inference(trial, data_path, path_ckpt, path_log, gpu, max_len, batch_size, batch_size_valid,
              dropout, emb_dim, heads, layers, att_dim, mlp_dim, warmup, lbl_smooth, eos, bos, epochs):

    ############# PICK GPU ###################
    if gpu != None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    # form paths
    path_ger_vocab = pform(data_path, "vocab_ger.model")
    path_eng_vocab = pform(data_path, "vocab_eng.model")
    path_validtxt = pform(data_path, "valid_txt.npy")
    path_validtgt = pform(data_path, "valid_tgt.npy")

    path_ckpt = expanduser(path_ckpt)
    path_log = expanduser(path_log)

    # sentence piece model
    vocab_enc = sp.load_spm(path_ger_vocab)
    vocab_dec = sp.load_spm(path_eng_vocab)
    dim_vocab = vocab_enc.GetPieceSize()


    #validation data
    valid_txt = np.load(path_validtxt)
    valid_tgt = np.load(path_validtgt)

    # dummy data to load graph
    data = np.zeros((2,2,2))

    # load graph
    model = transformer(data, batch_size, emb_dim, heads, layers, att_dim, mlp_dim,
                        dim_vocab, max_len, lbl_smooth, eos, dropout, False, warmup)

    # restore parameters
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(sess, pform(path_ckpt, trial))

    sent_ids = []
    for i, j in tqdm(partition(len(valid_txt), batch_size_valid, discard=False)):
        enc_output, mask = sess.run((model['enc_output'], model['mask_pad']),
                                    {model['enc_inpt']: vpack(valid_txt[i:j], (batch_size_valid, max(map(len, valid_txt[i:j]))), eos, np.int32)})
        sent = np.ones((len(enc_output), 1), dtype=np.int)*bos
        for _ in range(max_len):
            pred = sess.run(model['pred'], {model['enc_output']: enc_output,
                                            model['dec_inpt']: sent,
                                            model['mask_pad']: mask,
                                            model['train']: False})
            sent = np.concatenate([sent, pred[:,[-1]]], axis=1)
            if all(sent[:, -1] == eos):
                break
        sent_ids.extend(sent)

    pred_sents = [vocab_dec.decode_ids(sent.tolist()) for sent in sent_ids]
    valid_sents = [vocab_dec.decode_ids(sent.tolist()) for sent in valid_tgt]

    save_txt("../results/tf_norm_pred.txt", pred_sents)
    save_txt("../results/testset.txt", valid_sents)


if __name__ == "__main__":
    for key, val in params.items():
        exec(key+"=val")
    inference(trial, data_path, path_ckpt, path_log, gpu, max_len, batch_size, batch_size_valid,
              dropout, emb_dim, heads, layers, att_dim, mlp_dim, warmup, lbl_smooth, eos, bos, epochs)
