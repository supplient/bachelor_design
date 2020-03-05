import os

def seq2str(seq):
    s = ""
    for c in seq:
        s += c
    return s

def str2seq(s):
    return [c for c in s]

def load_file(input_file):
    if not os.path.exists(input_file):
        raise Exception(input_file + " doesn't exist.")

    char_seqs = []
    tag_seqs = []
    char_seq = [] # e.g. 1选择一条保电人员信息记录
    tag_seq = [] # e.g. O O O O O B-EI I-EI B-EI I-EI B-EI I-EI O O
    with open(input_file) as fd:
        for line in fd:
            if line.strip() == "]":
                char_seqs.append(char_seq)
                char_seq = []
                tag_seqs.append(tag_seq)
                tag_seq = []
                continue

            splited_line = line.split(" ")
            if len(splited_line) != 2:
                continue
            char_seq.append(splited_line[0].strip())
            tag_seq.append(splited_line[1].strip())
    
    return char_seqs, tag_seqs


def shuffle_twin(a, b):
    if len(a) != len(b):
        raise Exception("Can only shuffle twin arrays with the same length.")

    index_list = [i for i in range(len(a))]
    import random
    random.shuffle(index_list)

    na = []
    nb = []
    for i in index_list:
        na.append(a[i])
        nb.append(b[i])

    return na, nb


def preprocess(char_seqs, tag_seqs, vocab_file, SEQ_LEN=512, cased=True, tag_vocab=None, TAG_PAD=''):
    from keras_bert import load_vocabulary, Tokenizer, TOKEN_CLS, TOKEN_SEP, TOKEN_UNK, TOKEN_PAD
    from keras.utils.np_utils import to_categorical

    # Load vocab & Init Tokenizer
    vocab = load_vocabulary(vocab_file)
    tokenizer = Tokenizer(vocab, cased=cased)

    # Tokenization
    token_seqs = []
    orig2token_maps = []
    for char_seq in char_seqs:
        orig2token_map = [0]
        token_seq = [TOKEN_CLS]
        for c in char_seq:
            orig2token_map.append(len(token_seq))
            tokens = tokenizer.tokenize(c)
            tokens = tokens[1: -1]
            token_seq.extend(tokens)
        orig2token_map.append(len(token_seq))
        token_seq.append(TOKEN_SEP)
        orig2token_maps.append(orig2token_map)
        token_seqs.append(token_seq)

    # token => token_id
    TAG_PAD_ID = 0
    token_id_seqs = []
    unk_id = vocab.get(TOKEN_UNK)
    for token_seq, orig2token_map in zip(token_seqs, orig2token_maps):
        token_id_seq = []
        for i in orig2token_map:
            token = token_seq[i]
            token_id = vocab.get(token, unk_id)
            token_id_seq.append(token_id)
        token_id_seqs.append(token_id_seq)

    # tag => tag_id
    # Reserve 0 for padding
    if tag_vocab != None:
        if tag_vocab[TAG_PAD] != TAG_PAD_ID:
            raise Exception("tag_vocab[" + TAG_PAD + "] must equals " + str(TAG_PAD_ID))
    else:
        tag_vocab = {TAG_PAD:TAG_PAD_ID}

    tag_id_seqs = []
    for tag_seq in tag_seqs:
        tag_id_seq = [TAG_PAD_ID] # for [CLS]
        for tag in tag_seq:
            if not tag_vocab.get(tag):
                tag_vocab[tag] = len(tag_vocab)
            tag_id_seq.append(tag_vocab[tag])
        tag_id_seq.append(TAG_PAD_ID) # for [SEP]
        tag_id_seqs.append(tag_id_seq)

    # padding
    TOKEN_PAD_ID = 0
    def padding(seq, pad_char):
        if len(seq) > SEQ_LEN:
            del seq[SEQ_LEN:]
        while len(seq) < SEQ_LEN:
            seq.append(pad_char)
    for token_id_seq in token_id_seqs:
        padding(token_id_seq, TOKEN_PAD_ID)
    for tag_id_seq in tag_id_seqs:
        padding(tag_id_seq, TAG_PAD_ID)

    # make tag_id one-hot
    one_hot_tag_id_seqs = []
    for tag_id_seq in tag_id_seqs:
        one_hot_tag_id_seqs.append(
            to_categorical(tag_id_seq, num_classes=len(tag_vocab))
            )

    # create segment_seqs
    segment_seqs = []
    for token_id_seq in token_id_seqs:
        segment_seq = [0] * len(token_id_seq)
        segment_seqs.append(segment_seq)


    return token_id_seqs, segment_seqs, one_hot_tag_id_seqs, tag_vocab


if __name__ == "__main__":
    # file_path = "/mnt/d/My Drive/Graduation/Data/train_dict.txt"
    file_path = "test.txt"
    vocab_path = "/mnt/d/My Drive/Graduation/BERT/multi_cased_L-12_H-768_A-12/vocab.txt"
    char_seqs, tag_seqs = load_file(file_path)
    char_seqs, tag_seqs = shuffle_twin(char_seqs, tag_seqs)
    token_id_seqs, segment_seqs, one_hot_tag_id_seqs, tag_vocab = preprocess(
        char_seqs, 
        tag_seqs, 
        vocab_path,
        SEQ_LEN=15
        )
    print(token_id_seqs, segment_seqs, one_hot_tag_id_seqs, tag_vocab)
    preprocess(char_seqs, tag_seqs, vocab_path, tag_vocab=tag_vocab)