import numpy as np

def analyze(char_seqs, tag_seqs, output_tag_id_seqs, 
        tag_vocab,
        verbose=0 # 0, 1, 2, the larger, the detailer
        ):
    # Reverse tag_vocab, we need id => tag here
    reversed_tag_vocab = {}
    for key, value in tag_vocab.items():
        reversed_tag_vocab[value] = key

    sent_count = 0
    char_count = 0
    wrong_count = 0
    for char_seq, tag_seq, output_tag_id_seq in zip(char_seqs, tag_seqs, output_tag_id_seqs):
        if verbose >= 1:
            print("-"*15)
            print("Sentence " + str(sent_count))
        output_seq = output_tag_id_seq[1:-1] # Remove [CLS] and [SEP]'s id
        for c, expect_tag, output_tag_id_one_hot in zip(char_seq, tag_seq, output_seq):
            output_tag_id = np.argmax(output_tag_id_one_hot)
            output_tag = reversed_tag_vocab[output_tag_id]

            if verbose >= 2:
                print(c + "\t" + expect_tag + "\t" + output_tag)

            char_count += 1
            if expect_tag != output_tag:
                if verbose == 1:
                    print(c + "\t" + expect_tag + "\t" + output_tag)
                wrong_count += 1
        sent_count += 1

    print("-"*15)
    print("All: " + str(char_count))
    print("Wrong: " + str(wrong_count))
    print("Wrong Rate: " + str(int(wrong_count/char_count * 100)) + "%")

if __name__ == "__main__":
    from preprocess import load_file, preprocess
    char_seqs, tag_seqs = load_file("test.txt")
    token_id_seqs, one_hot_tag_id_seqs, tag_vocab = preprocess(char_seqs, tag_seqs)
    analyze(char_seqs, tag_seqs, one_hot_tag_id_seqs, tag_vocab, 2)