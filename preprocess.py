import os

def seq2str(seq):
    s = ""
    for c in seq:
        s += c
    return s

def str2seq(s):
    return [c for c in s]
    
def padding(seq, pad_char, SEQ_LEN):
    if len(seq) > SEQ_LEN:
        del seq[SEQ_LEN:]
    while len(seq) < SEQ_LEN:
        seq.append(pad_char)


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


def tokenize(char_seqs, vocab, cased):
    from keras_bert import Tokenizer, TOKEN_CLS, TOKEN_SEP
    tokenizer = Tokenizer(vocab, cased=cased)

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

    return token_seqs, orig2token_maps



def preprocess_char(char_seqs, vocab, SEQ_LEN, cased):
    from keras_bert import TOKEN_UNK
    # Tokenization
    token_seqs, orig2token_maps = tokenize(char_seqs, vocab, cased)

    # token => token_id
    token_id_seqs = []
    unk_id = vocab.get(TOKEN_UNK)
    for token_seq, orig2token_map in zip(token_seqs, orig2token_maps):
        token_id_seq = []
        for i in orig2token_map:
            token = token_seq[i]
            token_id = vocab.get(token, unk_id)
            token_id_seq.append(token_id)
        token_id_seqs.append(token_id_seq)

    # padding
    TOKEN_PAD_ID = 0
    for token_id_seq in token_id_seqs:
        padding(token_id_seq, TOKEN_PAD_ID, SEQ_LEN)

    return token_id_seqs


def create_segment(seqs_len, seq_len):
    return [ [0]*seq_len ] * seqs_len


def preprocess_tag(tag_seqs, SEQ_LEN, tag_vocab=None, TAG_PAD=''):
    from keras.utils.np_utils import to_categorical
    
    TAG_PAD_ID = 0
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
    for tag_id_seq in tag_id_seqs:
        padding(tag_id_seq, TAG_PAD_ID, SEQ_LEN)

    # # make tag_id one-hot: when using softmax, this is needed
    # one_hot_tag_id_seqs = []
    # for tag_id_seq in tag_id_seqs:
        # one_hot_tag_id_seqs.append(
            # to_categorical(tag_id_seq, num_classes=len(tag_vocab))
            # )
    
    # expand dim: This is for CRF, since it requires shape 1 at dim 3
    for tag_id_seq in tag_id_seqs:
        for i in range(len(tag_id_seq)):
            tag_id_seq[i] = [tag_id_seq[i]]

    return tag_id_seqs, tag_vocab



def preprocess(char_seqs, tag_seqs, vocab_file, SEQ_LEN=512, cased=True, tag_vocab=None, TAG_PAD=''):
    from keras_bert import load_vocabulary

    # Load vocab
    vocab = load_vocabulary(vocab_file)

    # preprocess char_seqs
    token_id_seqs = preprocess_char(
        char_seqs,
        vocab, 
        SEQ_LEN, 
        cased
    )

    # create segment_seqs
    segment_seqs = create_segment(
        len(token_id_seqs),
        len(token_id_seqs[0])
    )

    # preprocess tag_seqs
    tag_id_seqs, tag_vocab = preprocess_tag(
        tag_seqs, 
        SEQ_LEN, 
        tag_vocab, 
        TAG_PAD
    )

    return token_id_seqs, segment_seqs, tag_id_seqs, tag_vocab


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