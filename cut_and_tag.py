import jieba

def load_stopwords(stop_words_file):
    return [line.strip() for line in open(stop_words_file, "r").readlines()]

def cal_word_edit_dist(lhs, rhs):
    ''' Calculate word edit distance from lhs to rhs. Just copied.
    '''
    if lhs == rhs:
        return 0

    len_1 = len(lhs)
    len_2 = len(rhs)
    dist_table = [[0] * (len_2 + 1) for i in range(len_1 + 1)]
    for i in range(len_1 + 1):
        dist_table[i][0] = i
    for j in range(len_2 + 1):
        dist_table[0][j] = j
    for i in range(1, len_1 + 1):
        for j in range(1, len_2 + 1):
            if lhs[i - 1] == rhs[j - 1]:
                cost = 0
            else:
                cost = 1
            deletion = dist_table[i - 1][j] + 1
            insertion = dist_table[i][j - 1] + 1
            substitution = dist_table[i - 1][j - 1] + cost
            dist_table[i][j] = min(min(deletion, insertion), substitution)
    return dist_table[len_1][len_2]

def cut_and_remove_stopwords(s, stopwords):
    s_cut = [x for x in jieba.cut(s) if not x in stopwords]
    return s_cut

def cut_and_tag_each_line(line, stopwords):
    # Split line
    line_splited = line.split("\t")
    if len(line_splited) < 3:
        return None # This line is an empty line
    sentence = line_splited[0].strip()
    count_name = line_splited[1].strip()
    count_tag = line_splited[2].strip()

    # Upper count label
    count_tag = count_tag.upper()

    # Cut sentence, count_name & Remove stopwords
    sentence_cut = cut_and_remove_stopwords(sentence, stopwords)
    count_name_cut = cut_and_remove_stopwords(count_name, stopwords)

    # Search words in sentence which is similiar with words in count_name
    char_seq = []
    tag_seq = []
    for sentence_word in sentence_cut:
        # Check whether similiar with any word in count_name
        word_tag = "O"
        for count_word in count_name_cut:
            dist = cal_word_edit_dist(sentence_word, count_word)
            min_len = min(len(sentence_word), len(count_word))

            tmp = (dist==0)
            tmp = tmp or (min_len<=3 and dist<2)
            tmp = tmp or (4<=min_len and min_len<=7 and dist<3)
            tmp = tmp or (8<=min_len and dist<4)

            # If similiar, assign count_name's tag to this word
            if tmp:
                word_tag = count_tag
                break
            
        # Tag this word char by char
        first_char_flag = True
        for c in sentence_word:
            char_tag = None
            if word_tag == "O":
                char_tag = "O"
            elif first_char_flag:
                char_tag = "B-" + word_tag
                first_char_flag = False
            else:
                char_tag = "I-" + word_tag
            tag_seq.append(char_tag)

            # Build char_seq for corresponding with tag_seq
            char_seq.append(c)
    return [sentence_cut, char_seq, tag_seq]


def cut_and_tag(input_file, stopwords_file):
    stopwords = load_stopwords(stopwords_file)
    stopwords.extend(["\t", "", " "]) # Add space because space to stop words' list

    cut_sen_seqs = []
    char_seqs = []
    tag_seqs = []
    with open(input_file, "r") as fd:
        for line in fd:
            ret = cut_and_tag_each_line(line, stopwords)
            if ret == None:
                continue
            sentence_cut, char_seq, tag_seq = ret

            cut_sen_seqs.append(sentence_cut)
            char_seqs.append(char_seq)
            tag_seqs.append(tag_seq)
    return [cut_sen_seqs, char_seqs, tag_seqs]

if __name__ == "__main__":
    from driver_amount import addh
    import config
    [cut_sen_seqs, char_seqs, tag_seqs] = cut_and_tag(
        "cut_and_tag_test_file.txt", 
        addh + config.STOPWORDS_PATH
    )

    print(str(cut_sen_seqs) + "\n")
    print(str(char_seqs) + "\n")
    print(str(tag_seqs) + "\n")




                

