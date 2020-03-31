import keras
import logging
import sys
import json

class EpochCheckpoint(keras.callbacks.Callback):
    def __init__(self, modelpath, logpath, period):
        if period < 0 or type(period) != int:
            raise Exception("period must be a positive integer")

        self.modelpath = modelpath
        self.logpath = logpath
        self.period = period

        self.epoch_count = 0
        self.train_rec = []

    def on_train_begin(self, logs={}):
        self.epoch_count = 0
        self.train_rec = [
            {
                "model_path": self.modelpath,
                "save_period": self.period
            }
        ]

        return super().on_train_begin(logs=logs)

    def on_epoch_end(self, epoch, logs={}):
        self.epoch_count += 1

        if self.epoch_count < self.period:
            return super().on_epoch_end(epoch, logs=logs)
        self.epoch_count = 0

        # Save checkpoint
        self.model.save(self.modelpath)

        # Update record
        self.train_rec.append(
            {
                "epoch": epoch,
                "logs": logs
            }
        )
        with open(self.logpath, "w") as fd:
            json.dump(self.train_rec, fd)
            
        return super().on_epoch_end(epoch, logs=logs)
