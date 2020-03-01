# address head
COLAB_AMOUNT_PATH = "/gdrive"
COLAB_ADDH = COLAB_AMOUNT_PATH+ "/My Drive"
LOCAL_ADDH = "/mnt/d/My Drive"

# input file path
TRAIN_DATA_PATH = "/Graduation/Data/train_dict.txt"
TEST_DATA_PATH = "/Graduation/Data/test_dict.txt"

BERT_PATH = "/Graduation/BERT/multi_cased_L-12_H-768_A-12"
BERT_VOCAB_PATH = BERT_PATH + "/vocab.txt"
BERT_CONFIG_PATH = BERT_PATH + "/bert_config.json"
BERT_CHECKPOINT_PATH = BERT_PATH + "/bert_model.ckpt"

# generated file path
TAG_VOCAB_PATH = "/Graduation/Model/tag_vocab.pkl"
MODEL_PATH = "/Graduation/Model/model.h5"

# train & test common config
    # Configs which are specialized 
    # when training or testing will not be listed here
SEQ_LEN = 512