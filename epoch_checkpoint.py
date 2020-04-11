import keras
import logging
import sys
import json
import numpy as np
import os

def getOriginTag(tag):
    if ("B-" in tag) or ("I-" in tag):
        return tag[2:]
    return ""

def judgeWhichTag(tag_id_seq, rev_tag_vocab):
    for tag_id in tag_id_seq:
        tag = rev_tag_vocab[tag_id[0]]
        tag = getOriginTag(tag)
        if tag != "":
            return tag
    return "" # Does not belong any tag

class EpochCheckpoint(keras.callbacks.Callback):
    def __init__(self, modelpath, recpath, period, 
                test_input_seqs, test_tag_id_seqs, tag_vocab):
        if period < 0 or type(period) != int:
            raise Exception("period must be a positive integer")

        # Save input params
        self.modelpath = modelpath
        self.recpath = recpath
        self.period = period
        self.input_seqs = test_input_seqs
        self.tag_id_seqs = test_tag_id_seqs
        self.rev_tag_vocab = {} # Reverse tag_vocab, we need id => tag here
        for key, value in tag_vocab.items():
            self.rev_tag_vocab[value] = key

        # Init train_rec
        self.train_rec = []
        if os.path.exists(self.recpath):
            with open(self.recpath, "r") as fd:
                self.train_rec = json.load(fd)
                if self.train_rec[0]["completed"] == "Yes":
                    self.train_rec = []

        # Init runtime vars
        self.epoch_count = 0

        # Judge each seq's tag
        self.expect_tags = []
        for tag_id_seq in self.tag_id_seqs:
            self.expect_tags.append(
                judgeWhichTag(tag_id_seq, self.rev_tag_vocab)
            )


    def on_train_begin(self, logs={}):
        self.epoch_count = 0
        if len(self.train_rec) < 1:
            self.train_rec = [
                {
                    "model_path": self.modelpath,
                    "save_period": self.period,
                    "train_params": self.params,
                    "completed": "No",
                }
            ]
            self.updateTrainRec()

        return super().on_train_begin(logs=logs)

    def on_train_end(self, logs={}):
        self.train_rec[0]["completed"] = "Yes"
        self.updateTrainRec()
        return super().on_train_end(logs=logs)

    def on_epoch_end(self, epoch, logs={}):
        self.epoch_count += 1

        for key, value in logs.items():
            logs[key] = float(value)

        if self.epoch_count < self.period:
            return super().on_epoch_end(epoch, logs=logs)
        self.epoch_count = 0

        # Save checkpoint
        print("Saving checkpint...")
        self.model.save(self.modelpath)

        # Do predict
        print("Predicing for matrics calculating...")
        output_one_hot_seqs = self.model.predict(
            self.input_seqs,
            batch_size=self.params["batch_size"],
            verbose=1
        )
        output_seqs = []
        for one_hot_seq in output_one_hot_seqs:
            one_hot_seq = one_hot_seq[1:-1] # [1:-1] is to remove [CLS] and [SEP]
            seq = [[np.argmax(x)] for x in one_hot_seq]
            output_seqs.append(seq)

        # Judge each output seq's tag
        output_tags = [judgeWhichTag(seq, self.rev_tag_vocab) for seq in output_seqs]

        # Update record
        print("Updating train record...")
        self.train_rec.append(
            {
                "epoch": epoch,
                "logs": logs,
                "expect_tags": self.expect_tags,
                "output_tags": output_tags,
            }
        )
        self.updateTrainRec()
            
        return super().on_epoch_end(epoch, logs=logs)

    def updateTrainRec(self):
        with open(self.recpath, "w") as fd:
            json.dump(self.train_rec, fd)