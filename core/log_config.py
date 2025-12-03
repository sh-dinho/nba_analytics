# ============================================================
# File: core/log_config.py
# Purpose: Standardized logging with console + rotating file
# ============================================================

import logging
from logging.handlers import RotatingFileHandler
from core.config import BASE_LOGS_DIR

def setup_logger(name: str, level=logging.INFO):
    """
    Create a logger with both console and rotating file handlers.
    Ensures consistent logging across all pipeline scripts.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # Rotating file handler
        log_file = BASE_LOGS_DIR / f"{name}.log"
        fh = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=5)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger