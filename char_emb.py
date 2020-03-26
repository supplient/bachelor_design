from driver_amount import addh
import config
import keras
import keras_bert

class CharEmbedder:
    methods = [
        "emb_layer",
        "second_to_last_layer",
        "last_layer",
        "sum_all",
        "sum_last_four",
        # TODO "concat_last_four"
    ]

    def __init__(self):
        self.bert_model, bert_model_config = keras_bert.build_model_from_config(
            addh + config.BERT_CONFIG_PATH,
            trainable=False
        )
        self.vocab = keras_bert.load_vocabulary(addh + config.BERT_VOCAB_PATH)
        self.model = None

    def embed(self, char_seqs, method_name="emb_layer"):
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

        # Select Method to Build Model
        method = getattr(
            CharEmbedder,
            "_method_" + method_name,
            None
            )
        if method == None:
            raise Exception("Invalid method name: " + method_name)
        method(self)

        # Model predicting
        predict_res = self.model.predict(
            [token_id_seqs, segment_seqs],
            batch_size=4
        )

        # Remove [CLS] & [SEP]
        emb_seqs = []
        for i in range(len(predict_res)):
            emb_seq = predict_res[i][1:-2]
            emb_seqs.append(emb_seq)

        # Select Method to do after-processing
        after_method = getattr(
            CharEmbedder,
            "_after_" + method_name,
            None
        )
        if after_method != None:
            emb_seqs = after_method(self, emb_seqs)

        return emb_seqs



    def _method_single_layer(self, layer_name):
        # Build Model
        output = self.bert_model.get_layer(layer_name).output
        self.model = keras.models.Model(self.bert_model.input, output)
        self.model.load_weights(addh + config.MODEL_PATH, by_name=True)

    def _method_emb_layer(self):
        self._method_single_layer(
            "Embedding-Norm"
        )

    def _method_second_to_last_layer(self):
        self._method_single_layer(
            "Encoder-11-FeedForward-Norm"
        )

    def _method_last_layer(self):
        self._method_single_layer(
            "Encoder-12-FeedForward-Norm"
        )

    def _method_sum_layers(self, layers_name):
        # Build Model
        middle_outputs = []
        for layer_name in layers_name:
            output = self.bert_model.get_layer(layer_name).output
            middle_outputs.append(output)
        output = keras.layers.Add()(middle_outputs)
        self.model = keras.models.Model(self.bert_model.input, output)
        self.model.load_weights(addh + config.MODEL_PATH, by_name=True)
    
    def _method_sum_last_four(self):
        layers = []
        for i in range(9, 13):
            layers.append("Encoder-%i-FeedForward-Norm" % i)
        self._method_sum_layers(
            layers
        )

    def _method_sum_all(self):
        layers = []
        for i in range(1, 13):
            layers.append("Encoder-%i-FeedForward-Norm" % i)
        self._method_sum_layers(
            layers
        )

    def _after_normalization(self, emb_seqs):
        '''Normalizaion emb_seqs.
        Note: This function will edit emb_seqs
        '''
        import numpy as np
        for i in range(len(emb_seqs)):
            emb_seq = np.array(emb_seqs[i])
            mean = np.mean(emb_seq)
            std = np.std(emb_seq)
            emb_seqs[i] = [(x-mean)/std for x in emb_seq]
        return emb_seqs

    def _after_sum_last_four(self, emb_seqs):
        return self._after_normalization(emb_seqs)

    def _after_sum_all(self, emb_seqs):
        return self._after_normalization(emb_seqs)



    @classmethod
    def test(cls):
        import numpy as np

        a = [
            "给予生以希望", 
            "赐予死以冥道",
            "将希望交给生",
            "将冥道交给死"
        ]
        from preprocess import str2seq
        a = [str2seq(s) for s in a]

        embedder = CharEmbedder()
        for method in CharEmbedder.methods:
            a_emb = embedder.embed(a, method)
            print(method + ": ")
            p = np.array(a_emb)
            print(p.shape)


if __name__ == "__main__":
    CharEmbedder.test()