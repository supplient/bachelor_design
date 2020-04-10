# input file path
DATA_PATH = "/Graduation/Data/origin_data.txt"

STOPWORDS_PATH = "/Graduation/Data/stopwords.txt"

BERT_PATH = "/Graduation/BERT/multi_cased_L-12_H-768_A-12"
BERT_VOCAB_PATH = BERT_PATH + "/vocab.txt"
BERT_CONFIG_PATH = BERT_PATH + "/bert_config.json"
BERT_CHECKPOINT_PATH = BERT_PATH + "/bert_model.ckpt"

# generated file path
TAG_VOCAB_PATH = "/Graduation/Model/tag_vocab.json"
MODEL_PATH = "/Graduation/Model/model.h5"
EQUAL_DATA_PATH = "/Graduation/Data/equal.json"
EQUAL_PARAM_PATH = "/Graduation/Model/equal_param.json"

# log file path
TRAIN_REC_PATH = "/Graduation/Log/train.json" 
TRAIN_TABLE_PATH = "/Graduation/Log/train_table.csv"
EQUAL_TRAIN_REC_PATH = "/Graduation/Log/equal_train_rec.json"
EQUAL_TRAIN_REP_PATH = "/Graduation/Log/equal_train_report.txt"
EQUAL_TRAIN_TABLE_PATH = "/Graduation/Log/equal_train_table.csv"
EQUAL_SIF_TRAIN_REC_PATH = "/Graduation/Log/equal_sif_train_rec.json"
EQUAL_SIF_TRAIN_REP_PATH = "/Graduation/Log/equal_sif_train_rep.json"

# train & test common config
    # Configs which are specialized 
    # when training or testing will not be listed here
SEQ_LEN = 512

# util
LOG_PATH = "/Graduation/Model/system.log" # To move