# core/logging.py
import logging

def setup_logger(name: str, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger
