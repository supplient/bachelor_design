import config

def cal_dis(a, b):
    # TODO This is wrong, because each sentence is not a vector, but a 512X768 matrix
    # Here I need more survey
    if len(a) != len(b):
        raise Exception("Can only cal vecs with the same length.")
    res = 0
    for ai, bi in zip(a, b):
        res += abs(ai-bi)**2
    # Ignore sqrt since we only compare
    return res 

class Embedder:
    def __init__(self):
        import keras
        import keras_bert
        from driver_amount import addh

        bert_model, bert_model_config = keras_bert.build_model_from_config(
            addh + config.BERT_CONFIG_PATH,
            trainable=False
        )
        output = bert_model.get_layer("Embedding-Norm").output

        self.model = keras.models.Model(bert_model.input, output)
        self.model.load_weights(addh + config.MODEL_PATH, by_name=True)
        self.vocab = keras_bert.load_vocabulary(addh + config.BERT_VOCAB_PATH)

    def embed(self, char_seqs):
        import preprocess

        token_id_seqs = preprocess.preprocess_char(
            char_seqs,
            self.vocab,
            config.SEQ_LEN,
            True
        )

        segment_seqs = preprocess.create_segment(
            len(token_id_seqs),
            len(token_id_seqs[0])
        )

        res = self.model.predict(
            [token_id_seqs, segment_seqs],
            batch_size=4
        )
        return res

    @classmethod
    def test(cls):
        a = ["给予生以希望", "赐予死以冥道"]
        b = ["将希望交给生", "将冥道交给死"]
        from preprocess import str2seq
        a = [str2seq(s) for s in a]
        b = [str2seq(s) for s in b]

        embedder = Embedder()
        a_embedded = embedder.embed(a)
        b_embedded = embedder.embed(b)
        print(a_embedded)
        print(cal_dis(a_embedded, b_embedded))


if __name__ == "__main__":
    em = Embedder()
    a = em.embed(["不吃饭"])
    print(len(a))
    print(len(a[0]))
    print(len(a[0][0]))