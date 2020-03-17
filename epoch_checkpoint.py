import keras
import logging
import sys

class EpochCheckpoint(keras.callbacks.Callback):
    def __init__(self, modelpath, logpath, period):
        if period < 0 or type(period) != int:
            raise Exception("period must be a positive integer")

        self.modelpath = modelpath
        self.logpath = logpath
        self.period = period

        # Set logger
        self.logger = logging.getLogger("EpochCheckpoint_" + logpath)
        self.logger.setLevel(logging.DEBUG)

        fh = logging.FileHandler(logpath, mode="w")
        fh.setLevel(logging.DEBUG)
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.INFO)

        formatter = logging.Formatter("%(asctime)s-[%(levelname)s] %(message)s")
        fh.setFormatter(formatter)
        sh.setFormatter(formatter)
        self.logger.addHandler(fh)
        self.logger.addHandler(sh)

    def on_train_begin(self, logs={}):
        self.epoch_count = 0
        self.logger.info("Start Train. Checkpoint will be save at " + self.modelpath)
        self.logger.info("Detail log file will be saved at " + self.logpath)

        return super().on_train_begin(logs=logs)

    def on_epoch_end(self, epoch, logs={}):
        self.logger.debug("Epoch " + str(epoch))
        self.epoch_count += 1
        if self.epoch_count < self.period:
            return super().on_epoch_end(epoch, logs=logs)
        
        self.epoch_count = 0
        self.logger.info("Saving at epoch " + str(epoch) + "...")

        # Save checkpoint
        self.model.save(self.modelpath)
        self.logger.info("Save succeed.")
            
        return super().on_epoch_end(epoch, logs=logs)
