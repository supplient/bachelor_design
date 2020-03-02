import keras
import logging
import config
import sys

def getLogger(addh):
    if _getLogger.logger == None:
        _getLogger.init(addh)
    return _getLogger.logger

class _getLogger(object):
    logger = None
    @classmethod
    def init(cls, addh):
        if _getLogger.logger != None:
            return _getLogger.logger
        _getLogger.logger = logging.getLogger("System")
        _getLogger.logger.setLevel(logging.DEBUG)

        fh = logging.FileHandler(addh + config.LOG_PATH, mode="w")
        fh.setLevel(logging.DEBUG)
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.INFO)

        formatter = logging.Formatter("%(asctime)s-[%(levelname)s] %(message)s")
        fh.setFormatter(formatter)
        sh.setFormatter(formatter)

        _getLogger.logger.addHandler(fh)
        _getLogger.logger.addHandler(sh)

class EpochCheckpoint(keras.callbacks.Callback):
    def EpochCheckpoint(self, modelpath, logpath, period):
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
        formatter = logging.Formatter("%(asctime)s-[%(levelname)s] %(message)s")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def on_train_begin(self, logs={}):
        self.epoch_count = 0
        self.logger.info("Start Train. Checkpoint will be save at " + self.modelpath)

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



if __name__ == "__main__":
    from driver_amount import addh
    logger = getLogger(addh)
    logger.info("233")
    logger.debug("123")