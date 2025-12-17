# ============================================================
# File: src/utils/common.py
# Purpose: Common utilities (logging, IO) for pipeline v2.0
# Version: 2.0
# Author: Your Team
# Date: December 2025
# ============================================================

import logging
from pathlib import Path
import pandas as pd


def configure_logging(name="pipeline", level="INFO"):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger


def save_dataframe(df: pd.DataFrame, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    elif path.suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported file type: {path}")


def load_dataframe(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    elif path.suffix == ".csv":
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates()
