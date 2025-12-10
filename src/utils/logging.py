# ============================================================
# File: src/utils/logging.py
# Purpose: Configure logging
# ============================================================

import logging
from pathlib import Path

def configure_logging(name="app", level="INFO", log_dir="logs"):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        fh = logging.FileHandler(f"{log_dir}/{name}.log")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger
