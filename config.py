# input file path
DATA_PATH = "/Graduation/Data/origin_data.txt"

STOPWORDS_PATH = "/Graduation/Data/stopwords.txt"

BERT_PATH = "/Graduation/BERT/multi_cased_L-12_H-768_A-12"
BERT_VOCAB_PATH = BERT_PATH + "/vocab.txt"
BERT_CONFIG_PATH = BERT_PATH + "/bert_config.json"
BERT_CHECKPOINT_PATH = BERT_PATH + "/bert_model.ckpt"

# generated file path
TAG_VOCAB_PATH = "/Graduation/Model/tag_vocab.pkl"
MODEL_PATH = "/Graduation/Model/model.h5"
TRAIN_LOG_PATH = "/Graduation/Model/train.log" # This should be in Model, it's right.
EQUAL_DATA_PATH = "/Graduation/Data/equal.json"
EQUAL_PARAM_PATH = "/Graduation/Model/equal_param.json"
EQUAL_TRAIN_REC_PATH = "/Graduation/Model/equal_train_rec.json" # To move

# log file path
EQUAL_TRAIN_REP_PATH = "/Graduation/Log/equal_train_report.txt"
EQUAL_SIF_TRAIN_REC_PATH = "/Graduation/Log/equal_sif_train_rec.json"

# train & test common config
    # Configs which are specialized 
    # when training or testing will not be listed here
SEQ_LEN = 512

# util
LOG_PATH = "/Graduation/Model/system.log" # To move