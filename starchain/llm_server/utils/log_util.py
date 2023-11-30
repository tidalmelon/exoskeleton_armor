import os
import re
import datetime
import logging
from logging.handlers import TimedRotatingFileHandler


def get_logger(log_file_name, log_level=logging.INFO, time_level="D", rollovernow=True):
    fmt = '%(asctime)s\tFile \"%(filename)s\",line %(lineno)s\t%(levelname)s: %(message)s'
    fmt = logging.Formatter(fmt)

    # console handler
    handler_console = logging.StreamHandler()
    handler_console.setFormatter(fmt)

    # file handler
    handler_file = TimedRotatingFileHandler(filename=log_file_name, when=time_level, interval=1, backupCount=3)
    # handler_file.suffix = "%Y-%m-%d_%H-%M.log"
    # handler_file.extMatch = re.compile(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}.log")
    handler_file.suffix = "%Y-%m-%d.log"
    handler_file.extMatch = re.compile(r"\d{4}-\d{2}-\d{2}.log")
    handler_file.setFormatter(fmt)
    # 首先触发 TimedRotatingFileHandler 删除日志
    if rollovernow:
        # baseFilename 'D:\\project\\daily\\lossdata.log'
        # suffer '%Y-%m-%d.log'

        relative_name = datetime.datetime.now().strftime(handler_file.suffix)
        absolute_name = handler_file.baseFilename + f'.{relative_name}'

        log_today_exists = os.path.exists(absolute_name)
        if not log_today_exists:
            handler_file.doRollover()

    logging.basicConfig(level=log_level)

    logger = logging.getLogger()

    # clear origin handler
    logger.handlers.clear()

    logger.addHandler(handler_console)
    logger.addHandler(handler_file)
    return logger
