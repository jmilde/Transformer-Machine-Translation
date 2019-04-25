from util_io import save_txt, pform
from util_sp import load_spm, spm
from util_np import np, unison_shfl
from hyperparameters import preprocess_params as params


def preprocess(data_path, valid_size, max_len):

    # form path
    path_data_ger = pform(data_path, "de-en/europarl-v7.de-en.de")
    path_data_eng = pform(data_path, "de-en/europarl-v7.de-en.en")
    path_vocab_ger = pform(data_path, "vocab_ger")
    path_vocab_eng = pform(data_path, "vocab_eng")
    path_ger_txt = pform(data_path, "ger_txt.txt")
    path_eng_txt = pform(data_path, "eng_txt.txt")
    path_traintxt = pform(data_path, "train_txt.npy")
    path_traintgt = pform(data_path, "train_tgt.npy")
    path_validtxt = pform(data_path, "valid_txt.npy")
    path_validtgt = pform(data_path, "valid_tgt.npy")

    # load data
    data_ger = open(path_data_ger).read().split("\n")
    data_eng = open(path_data_eng).read().split("\n")


    # lowercase and remove empty lines
    avoid = set([idx for idx, sent in enumerate(zip(data_eng, data_ger))
                 if len(sent[0])==0 or len(sent[1])==0])

    sent_ger = [sent.lower() for idx, sent in enumerate(data_ger) \
                if idx not in avoid]
    sent_eng = [sent.lower() for idx, sent in enumerate(data_eng) \
                if idx not in avoid]

    # save text for sentence piece model
    save_txt(path_ger_txt, sent_ger)
    save_txt(path_eng_txt, sent_eng)

    # train sentence piece model
    spm(name=path_vocab_ger, path=path_ger_txt)
    spm(name=path_vocab_eng, path=path_eng_txt)

    # load trained sentence piece models
    vocab_ger = load_spm(path_vocab_ger + ".model")
    vocab_ger.SetEncodeExtraOptions("eos") # enable start/end symbols
    vocab_eng = load_spm(path_vocab_eng + ".model")
    vocab_eng.SetEncodeExtraOptions("bos:eos") # enable start/end symbols

    # shuffle
    sent_ger, sent_eng = unison_shfl(sent_ger, sent_eng)

    # encode with sentence piece and save validation set
    valid_txt, valid_tgt = [], []
    for idx, sent in enumerate(zip(sent_ger, sent_eng)):
        g = vocab_ger.encode_as_ids(sent[0])
        e = vocab_eng.encode_as_ids(sent[1])
        if len(g)<=max_len and len(e)<=max_len:
            valid_txt.append(g)
            valid_tgt.append(e)
            if len(valid_txt) == valid_size:
                last = idx
                break
    valid_txt = np.array([np.asarray(x, dtype=np.uint16) for x in valid_txt])
    valid_tgt = np.array([np.asarray(x, dtype=np.uint16) for x in valid_tgt])
    np.save(path_validtxt, valid_txt)
    np.save(path_validtgt, valid_tgt)

    # encode with sentence piece and save training set
    train_txt, train_tgt = [], []
    for ger, eng in zip(sent_ger[last:], sent_eng[last:]):
        g = vocab_ger.encode_as_ids(ger)
        e = vocab_eng.encode_as_ids(eng)
        if len(g)<=max_len and len(e)<=max_len:
            train_txt.append(g)
            train_tgt.append(e)

    train_txt = np.array([np.asarray(x, dtype=np.uint16) for x in train_txt])
    train_tgt = np.array([np.asarray(x, dtype=np.uint16) for x in train_tgt])
    np.save(path_traintxt, train_txt)
    np.save(path_traintgt, train_tgt)

if __name__ == "__main__":
    for key, val in params.items():
        exec(key+"=val")
    preprocess(data_path, valid_size, max_len)
