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

# log file path
TRAIN_REC_PATH = "/Graduation/Log/train.json" 
TRAIN_TABLE_PATH = "/Graduation/Log/train_table.csv"

# train & test common config
    # Configs which are specialized 
    # when training or testing will not be listed here
SEQ_LEN = 512

# util
LOG_PATH = "/Graduation/Model/system.log" # To move