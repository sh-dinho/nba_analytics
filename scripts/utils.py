# File: scripts/utils.py

import os
import logging
from datetime import datetime
import pandas as pd

def setup_logger(name, level=logging.INFO, log_to_file=None):
    """
    Setup a logger with console output and optional file output.

    Parameters:
        name (str): Logger name.
        level (int): Logging level (default: logging.INFO)
        log_to_file (str or None): Path to log file. If None, no file logging.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
        logger.addHandler(ch)

        # Optional file handler
        if log_to_file:
            os.makedirs(os.path.dirname(log_to_file), exist_ok=True)
            fh = logging.FileHandler(log_to_file)
            fh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
            logger.addHandler(fh)

    return logger


def get_timestamp(fmt="%Y%m%d_%H%M%S"):
    """
    Returns current timestamp string.
    
    Parameters:
        fmt (str): datetime format string.
    
    Returns:
        str: formatted timestamp
    """
    return datetime.now().strftime(fmt)


def ensure_columns(df: pd.DataFrame, required_cols, name="DataFrame"):
    """
    Check if required columns exist in a DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame to check.
        required_cols (list): List of column names to ensure.
        name (str): Optional name for error messages.
    
    Raises:
        ValueError: if any required column is missing.
    """
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")
