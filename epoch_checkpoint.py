import keras
import logging
import sys
import json
import numpy as np

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

        # Init runtime vars
        self.epoch_count = 0
        self.train_rec = []
        self.SMALL_CONST = 0.00000001

        # Judge each seq's tag
        self.tags = []
        for tag_id_seq in self.tag_id_seqs:
            self.tags.append(
                judgeWhichTag(tag_id_seq, self.rev_tag_vocab)
            )

        # Count each tag category's expect number
        self.expect_tag_num = {}
        for tag in self.tags:
            num = self.expect_tag_num.get(tag, 0)
            self.expect_tag_num[tag] = num+1


    def on_train_begin(self, logs={}):
        self.epoch_count = 0
        self.train_rec = [
            {
                "model_path": self.modelpath,
                "save_period": self.period,
                "train_params": self.params,
                "expect_tag_num": self.expect_tag_num
            }
        ]

        return super().on_train_begin(logs=logs)

    def on_epoch_end(self, epoch, logs={}):
        self.epoch_count += 1

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

        # Calculate metrics
        cost_prec, label_prec, output_tag_num = self.cal_metrics(output_seqs)

        # Update record
        self.train_rec.append(
            {
                "epoch": epoch,
                "logs": logs,
                "cost_precision": cost_prec,
                "label_precision": label_prec,
                "output_tag_num": output_tag_num
            }
        )
        with open(self.recpath, "w") as fd:
            json.dump(self.train_rec, fd, indent=2)
            
        print("-cost_precision: %f  -label_precision: %f" % (cost_prec, label_prec))
        return super().on_epoch_end(epoch, logs=logs)


    def cal_metrics(self, output_seqs):
        # Judge each output seq's tag
        output_tags = [judgeWhichTag(seq, self.rev_tag_vocab) for seq in output_seqs]

        # Cal cost precision
        ## Count each tag category's actual output num
        output_tag_num = {}
        for tag in output_tags:
            num = output_tag_num.get(tag, 0)
            output_tag_num[tag] = num + 1

        ## Cal each tag's cost precision
        cost_p = {}
        for tag in self.expect_tag_num.keys():
            if tag == "":
                continue
            expect = self.expect_tag_num.get(tag, 0)
            output = output_tag_num.get(tag, 0)
            
            diff = abs(expect - output)
            if expect == 0:
                cost_p[tag] = 0
            else:
                cost_p[tag] = max(0, expect-diff)/expect
        print(cost_p)

        ## Cal total cost precision
        total_cost_p = 1
        for tag, p in cost_p.items():
            total_cost_p += p
        total_cost_p /= len(cost_p)

        # Cal label precision
        right_count = 0
        for expect, output in zip(self.tags, output_tags):
            if expect == output:
                right_count += 1
        label_p = right_count/len(output_tags)

        return total_cost_p, label_p, output_tag_num
