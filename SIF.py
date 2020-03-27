import numpy as np

def cal_char_freq(cut_seqs):
    # Calculate word's frequency
    freq = {}
    for cut_seq in cut_seqs:
        for w in cut_seq:
            if not w in freq.keys():
                freq[w] = 0
            freq[w] += 1
    total = sum([count for w, count in freq.items()])
    for w, count in freq.items():
        freq[w] = freq[w] / total

    # Char's frequency inherit from word's frequency
    freq_seqs = []
    for cut_seq in cut_seqs:
        freq_seq = []
        for w in cut_seq:
            w_seq = [freq[w]] * len(w)
            freq_seq.extend(w_seq)
        freq_seqs.append(freq_seq)
    return freq_seqs

def cal_weight(freq, alpha):
    return alpha / (alpha * freq)

# Copy from https://github.com/PrincetonML/SIF/blob/master/src/SIF_embedding.py
def compute_pc(X,npc=1):
    """
    Compute the principal components. DO NOT MAKE THE DATA ZERO MEAN!
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: component_[i,:] is the i-th pc
    """
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
    svd.fit(X)
    return svd.components_

# Copy from https://github.com/PrincetonML/SIF/blob/master/src/SIF_embedding.py
def remove_pc(X, npc=1):
    """
    Remove the projection on the principal components
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: XX[i, :] is the data point after removing its projection
    """
    pc = compute_pc(X, npc)
    if npc==1:
        XX = X - X.dot(pc.transpose()) * pc
    else:
        XX = X - X.dot(pc.transpose()).dot(pc)
    return XX


class SIF:
    def __init__(self, alpha):
        self.alpha = alpha
        pass

    def compose(self, cut_seqs, char_seqs, emb_seqs):
        # Cal frequency
        freq_seqs = cal_char_freq(cut_seqs)

        # Compose meta sentence vector via char vector
        #       Here we turn list to numpy array for calculating convenience
        sen_num = len(char_seqs)
        emb_len = len(emb_seqs[0][0])

        sen_vecs = np.zeros((sen_num, emb_len))
        sen_index = 0
        for char_seq, emb_seq, freq_seq in zip(char_seqs, emb_seqs, freq_seqs):
            sen_vec = np.zeros((emb_len))
            
            # cal weighted mean
            for c, emb, freq in zip(char_seq, emb_seq, freq_seq):
                emb = np.array(emb)
                weight = cal_weight(freq, self.alpha)
                sen_vec = sen_vec + weight * emb

            # normalization
            sen_len = min(len(char_seq), len(emb_seq))
            sen_vec = sen_vec / sen_len
            sen_vecs[sen_index, :] = sen_vec

            sen_index += 1
        if sen_index != sen_num:
            raise Exception("Not all sentences are calculated successfully, only " + str(sen_index) + " succeeded")

        # Remove principal princess
        sen_vecs = remove_pc(sen_vecs)

        return sen_vecs.tolist()

    @classmethod
    def test(cls):
        cut_seqs = [
            ["a", "b"],
            ["a", "d"],
        ]
        char_seqs = [
            ["a", "b"],
            ["a", "d"],
        ]
        emb_seqs = [
            [
                [0, 1],
                [1, 0]
            ],
            [
                [0, 1],
                [1, 1]
            ]
        ]
        alpha = 0.0005

        sif = SIF(alpha)
        sen_vecs = sif.compose(cut_seqs, char_seqs, emb_seqs)
        print(sen_vecs)

if __name__ == "__main__":
    SIF.test()


        
        


            