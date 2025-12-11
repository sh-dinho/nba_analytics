# ============================================================
# File: src/utils/io.py
# Purpose: Utility functions for loading and saving DataFrames
# Project: nba_analysis
# Version: 1.5 (adds daily rotating file handler for audit logs)
#
# Dependencies:
# - os (standard library)
# - pandas
# - logging (standard library)
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

# Configure timed rotating file handler (rotate at midnight, keep 7 days of logs)
timed_handler = TimedRotatingFileHandler(
    LOG_FILE, when="midnight", interval=1, backupCount=7, encoding="utf-8"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),  # Console output
        timed_handler             # Daily rotating audit log file
    ],
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("io_utils")


def load_dataframe(path: str) -> pd.DataFrame:
    """
    Load a DataFrame from a CSV or Parquet file.

    Arguments:
    path -- Path to the file

    Returns:
    DataFrame -- Loaded DataFrame

    Raises:
    FileNotFoundError -- If the file does not exist
    ValueError -- If the file format is unsupported
    """
    if not os.path.exists(path):
        logger.error(f"File not found: {path}")
        raise FileNotFoundError(f"{path} does not exist")

    if path.endswith(".csv"):
        df = pd.read_csv(path)
    elif path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        logger.error(f"Unsupported file format for {path}")
        raise ValueError("Unsupported file format: must be CSV or Parquet")

    logger.info(f"Loaded DataFrame from {path} (shape: {df.shape})")
    return df


def save_dataframe(df: pd.DataFrame, path: str):
    """
    Save a DataFrame to a CSV or Parquet file.

    Arguments:
    df -- DataFrame to save
    path -- Path to the output file

    Raises:
    ValueError -- If the file format is unsupported
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if path.endswith(".csv"):
        df.to_csv(path, index=False)
    elif path.endswith(".parquet"):
        df.to_parquet(path, index=False)
    else:
        logger.error(f"Unsupported file format for {path}")
        raise ValueError("Unsupported file format: must be CSV or Parquet")

    logger.info(f"Saved DataFrame to {path} (shape: {df.shape})")
