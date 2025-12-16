# ============================================================
# File: src/utils/common.py
# Purpose: Centralized utilities (logging, IO, cleaning, IDs, mapping)
# ============================================================

import os
import logging
import shutil
import pandas as pd
from pathlib import Path
from typing import Dict, Any


# -----------------------------
# Logging
# -----------------------------

logger = logging.getLogger("utils.common")


def save_dataframe(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info(f"Data saved to {path} (rows={len(df)})")


def load_dataframe(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        logger.warning(f"File not found: {path}")
        return pd.DataFrame()
    df = pd.read_parquet(path)
    logger.info(f"Data loaded from {path} (rows={len(df)})")
    return df


def configure_logging(
    log_file: str = None,
    retention_days: int = None,
    name: str = "nba_analysis",
    level: str = None,
    log_dir: str = None,
) -> logging.Logger:
    """
    Configure a logger with file + console handlers.
    """
    log_dir = (
        log_dir
        or os.path.dirname(log_file or os.getenv("LOG_FILE", "logs/app.log"))
        or "logs"
    )
    os.makedirs(log_dir, exist_ok=True)
    log_file = log_file or os.path.join(log_dir, "app.log")

    logger = logging.getLogger(name)
    logger.setLevel(
        getattr(logging, level or os.getenv("LOG_LEVEL", "INFO"), logging.INFO)
    )

    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)

    # Optional retention
    if retention_days:
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=retention_days)
        for file in Path(log_dir).glob("*.log"):
            if pd.Timestamp(file.stat().st_mtime, unit="s") < cutoff:
                try:
                    file.unlink()
                except Exception:
                    pass

    return logger


# -----------------------------
# IO Utilities
# -----------------------------
def _check_format(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext in [".csv"]:
        return "csv"
    elif ext in [".parquet"]:
        return "parquet"
    raise ValueError(f"Unsupported file format: {ext}")


def save_dataframe(df: pd.DataFrame, path: str, **kwargs) -> None:
    fmt = _check_format(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        df.to_csv(path, index=False, **kwargs)
    else:
        df.to_parquet(path, index=False, **kwargs)


def load_dataframe(path: str, **kwargs) -> pd.DataFrame:
    fmt = _check_format(path)
    if fmt == "csv":
        return pd.read_csv(path, **kwargs)
    return pd.read_parquet(path, **kwargs)


def read_or_create(path: str, schema: Dict[str, Any]) -> pd.DataFrame:
    if Path(path).exists():
        return load_dataframe(path)
    return pd.DataFrame(columns=schema)


# -----------------------------
# Data Cleaning
# -----------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates and reset index."""
    return df.drop_duplicates().reset_index(drop=True)


def rename_columns(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    return df.rename(columns=mapping)


def prepare_game_data(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure GAME_DATE is datetime and TEAM_ID is int."""
    if "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    if "TEAM_ID" in df.columns:
        df["TEAM_ID"] = pd.to_numeric(df["TEAM_ID"], errors="coerce").astype("Int64")
    return df


# -----------------------------
# Unique ID Utilities
# -----------------------------
def add_unique_id(
    df: pd.DataFrame, cols: list, new_col: str = "unique_id"
) -> pd.DataFrame:
    """Concatenate specified columns into a unique identifier."""
    df[new_col] = df[cols].astype(str).agg("_".join, axis=1)
    return df


# -----------------------------
# Mapping Utilities
# -----------------------------
def map_ids(df: pd.DataFrame, column: str, mapping: Dict[str, Any]) -> pd.DataFrame:
    """Generic ID mapping."""
    df[column] = df[column].map(mapping).fillna(df[column])
    return df
