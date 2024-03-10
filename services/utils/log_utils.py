import logging

import colorlog


def get_logger(name: str):
    log_format = "%(asctime)s | %(log_color)s%(levelname)s%(reset)s | %(cyan)s%(name)s:%(lineno)s%(reset)s | %(log_color)s%(message)s%(reset)s"
    formatter = colorlog.ColoredFormatter(log_format)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.propagate = False

    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)

    return logger
