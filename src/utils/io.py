# ============================================================
# File: src/utils/io.py
# Purpose: Utility functions for loading and saving DataFrames
# Project: nba_analysis
# Version: 1.7 (adds read_or_create helper)
# ============================================================

import os
import pandas as pd
import logging
from logging.handlers import TimedRotatingFileHandler

# -----------------------------
# LOGGING CONFIGURATION
# -----------------------------
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "io_operations.log")
os.makedirs(LOG_DIR, exist_ok=True)

timed_handler = TimedRotatingFileHandler(
    LOG_FILE, when="midnight", interval=1, backupCount=7, encoding="utf-8"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), timed_handler],
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("io_utils")


def load_dataframe(path: str) -> pd.DataFrame:
    """Load a DataFrame from a CSV or Parquet file."""
    if not os.path.exists(path):
        logger.error("File not found: %s", path)
        raise FileNotFoundError(f"{path} does not exist")

    if path.endswith(".csv"):
        df = pd.read_csv(path)
    elif path.endswith(".parquet"):
        df = pd.read_parquet(path, engine="pyarrow")
    else:
        logger.error("Unsupported file format for %s", path)
        raise ValueError("Unsupported file format: must be CSV or Parquet")

    logger.info("Loaded DataFrame from %s (shape: %s)", path, df.shape)
    return df


def save_dataframe(df: pd.DataFrame, path: str) -> str:
    """Save a DataFrame to a CSV or Parquet file."""
    if not isinstance(df, pd.DataFrame):
        logger.error("save_dataframe expects a pandas DataFrame, got %s", type(df))
        raise TypeError("save_dataframe expects a pandas DataFrame")

    os.makedirs(os.path.dirname(path), exist_ok=True)

    if path.endswith(".csv"):
        df.to_csv(path, index=False)
    elif path.endswith(".parquet"):
        df.to_parquet(path, index=False, engine="pyarrow")
    else:
        logger.error("Unsupported file format for %s", path)
        raise ValueError("Unsupported file format: must be CSV or Parquet")

    logger.info("Saved DataFrame to %s (shape: %s)", path, df.shape)
    return path


def read_or_create(path: str, default_df: pd.DataFrame) -> pd.DataFrame:
    """
    Load a DataFrame if the file exists, otherwise save and return the default DataFrame.

    Arguments:
    path -- Path to the file
    default_df -- DataFrame to save if file does not exist

    Returns:
    DataFrame -- Loaded or newly created DataFrame
    """
    if os.path.exists(path):
        logger.info("File exists, loading: %s", path)
        return load_dataframe(path)
    else:
        logger.info("File not found, creating new: %s", path)
        save_dataframe(default_df, path)
        return default_df
