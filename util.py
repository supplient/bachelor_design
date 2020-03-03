import logging
import config
import sys

def getLogger(addh=None):
    if _getLogger.logger == None:
        if addh == None:
            raise Exception("Logger has not been initiliazed, addh must be assigned.")
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

if __name__ == "__main__":
    from driver_amount import addh
    logger = getLogger(addh)
    logger.info("233")
    logger.debug("123")