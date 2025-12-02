import os
import logging
from datetime import datetime

def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
        logger.addHandler(ch)
    return logger

def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_columns(df, required_cols, name="DataFrame"):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")
