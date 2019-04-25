parameters = {"trial": "transformer",
              "data_path": "../data",
              "path_ckpt": "../ckpt",
              "path_log": "../log",
              "gpu": None,  # if training on a gpu add the number here (terminal: nvidia-smi)
              "max_len": 65,  # max length of the sequence -> look at preprocess_params
              "batch_size": 2,
              "batch_size_valid": 100,
              "dropout": 0.1,  # percentage of dropped connections
              "emb_dim": 512,  # dimensions of the embedding layer
              "heads": 8,  # number of heads for the multihead attention
              "layers": 3,  # number of layers in encoder and decoder
              "att_dim": 64,  # dimension of the attention
              "mlp_dim": 512,  #size of layer 1 & 3, layer 2 is mlp_dim*4
              "warmup": 4000,  # how many warmup steps until the stepsize gets decreased
              "lbl_smooth": 0.1,  # amount of label smoothing
              "eos": 1,  # number that represents the end of a sentence
              "bos": 2,  # number that represents the beginning of a sentence
              "epochs": 10}  # number of epochs to train for

preprocess_params = {"data_path": "../data",
                    "valid_size": 4000,  # size of the validation set
                    "max_len": 65}  # maximum length of the sequence
