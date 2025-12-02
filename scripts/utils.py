# File: scripts/utils.py

import os
import logging
from datetime import datetime
import pandas as pd
from typing import Union, List, Set, Tuple, Dict, Any

def setup_logger(name: str, level: int = logging.INFO, log_to_file: str = None) -> logging.Logger:
    """
    Setup a logger with console output and optional file output.
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
            fh = logging.FileHandler(log_to_file, encoding="utf-8")
            fh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
            logger.addHandler(fh)

    return logger


def get_timestamp(fmt: str = "%Y%m%d_%H%M%S") -> str:
    """Returns current timestamp string."""
    return datetime.now().strftime(fmt)


def ensure_columns(df: pd.DataFrame, required_cols: Union[List[str], Set[str], Tuple[str]], name: str = "DataFrame") -> bool:
    """
    Check if required columns exist in a DataFrame.
    Raises ValueError if any required column is missing.
    """
    required = list(required_cols)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")
    return True


def append_pipeline_summary(summary: Dict[str, Any], summary_file: str = "results/pipeline_summary.csv") -> None:
    """
    Append a pipeline summary dictionary to a rolling CSV file.

    Parameters:
        summary (dict): Dictionary of metrics and metadata for one pipeline run.
        summary_file (str): Path to the rolling summary CSV file.
    """
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)

    # Ensure timestamp is present
    if "timestamp" not in summary:
        summary["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    df = pd.DataFrame([summary])

    if os.path.exists(summary_file):
        df.to_csv(summary_file, mode="a", header=False, index=False)
    else:
        df.to_csv(summary_file, index=False)