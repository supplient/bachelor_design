import keras

class EpochCheckpoint(keras.callbacks.Callback):
    def EpochCheckpoint(self, modelpath, logpath, period):
        if period < 0 or type(period) != int:
            raise Exception("period must be a positive integer")

        self.modelpath = modelpath
        self.logpath = logpath
        self.period = period

        # Set logger
        import logging
        self.logger = logging.getLogger("EpochCheckpoint_" + logpath)
        self.logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(logpath)
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