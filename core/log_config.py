# ============================================================
# File: core/log_config.py
# Purpose: Standardized logging with console + rotating file
# ============================================================

import os
import logging
from logging.handlers import RotatingFileHandler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)


def setup_logger(name: str, level=logging.INFO) -> logging.Logger:
    """
    Create a logger with both console and rotating file handlers.
    Ensures consistent logging across all pipeline scripts.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        # File handler
        log_file = os.path.join(LOG_DIR, f"{name}.log")
        fh = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=5)
        fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        fh.setLevel(level)

        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(name)s - %(levelname)s - %(message)s"))
        ch.setLevel(level)

        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger