import config

class CharEmbedder:
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

        # Preprocess for model predicting
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

        # Model predicting
        predict_res = self.model.predict(
            [token_id_seqs, segment_seqs],
            batch_size=4
        )

        # Remove [CLS]
        #       There is no need to remove [SEP], since the origin length can be obtained from char_seqs
        emb_seqs = []
        for i in range(len(predict_res)):
            emb_seq = predict_res[i][1:]
            emb_seqs.append(emb_seq)

        return emb_seqs

    @classmethod
    def test(cls):
        a = ["给予生以希望", "赐予死以冥道"]
        b = ["将希望交给生", "将冥道交给死"]
        from preprocess import str2seq
        a = [str2seq(s) for s in a]
        b = [str2seq(s) for s in b]

        embedder = CharEmbedder()
        a_emb = embedder.embed(a)
        b_emb = embedder.embed(b)
        print(a_emb)
        print(b_emb)


if __name__ == "__main__":
    CharEmbedder.test()