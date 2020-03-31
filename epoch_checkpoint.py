import keras
import logging
import sys
import json

def judgeWhichTag(tag_id_seq, rev_tag_vocab):
    for tag_id in tag_id_seq:
        tag = rev_tag_vocab[tag_id[0]]
        if "B-" in tag:
            return tag[2:]
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
        self.SMALL_CONST = 10^-7

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
                "train_params": self.params
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
        output_seqs = self.model.predict(
            self.input_seqs,
            batch_size=self.params["batch_size"],
            verbose=1
        )

        # Calculate metrics
        cost_prec, label_prec = self.cal_metrics(output_seqs)

        # Update record
        self.train_rec.append(
            {
                "epoch": epoch,
                "logs": logs,
                "cost_precision": cost_prec,
                "label_precision": label_prec
            }
        )
        with open(self.recpath, "w") as fd:
            json.dump(self.train_rec, fd)
            
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
        for tag, expect in self.expect_tag_num.items():
            if tag == "":
                continue
            output = output_tag_num[tag]
            
            diff = abs(expect - output)
            cost_p[tag] = max(0, expect-diff)/expect

        ## Cal total cost precision
        total_cost_p = 1
        for tag, p in cost_p.items():
            total_cost_p *= 1/max(p, self.SMALL_CONST)
        total_cost_p /= len(cost_p)
        total_cost_p = 1/total_cost_p

        # Cal label precision
        right_count = 0
        for expect, output in zip(self.tags, output_tags):
            if expect == output:
                right_count += 1
        label_p = right_count/len(output_tags)

        return total_cost_p, label_p
