# ============================================================
# File: src/utils/io.py
# Purpose: File I/O helpers for NBA analysis project with logging
# Project: nba_analysis
# ============================================================

import json
import shutil
from pathlib import Path
from typing import Any, Union

import pandas as pd

from src.utils.logging import configure_logging

# Configure a module-level logger
logger = configure_logging(level="INFO", log_dir="logs", name="io")


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists, return Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Ensured directory exists: {p}")
    return p


def save_dataframe(df: pd.DataFrame, path: Union[str, Path]) -> Path:
    """Save DataFrame to Parquet, CSV, or Excel based on extension. Returns path."""
    p = Path(path)
    ensure_dir(p.parent)
    try:
        if p.suffix == ".parquet":
            try:
                df.to_parquet(p, index=False)
            except ImportError as e:
                raise RuntimeError("Parquet support requires pyarrow or fastparquet") from e
        elif p.suffix == ".csv":
            df.to_csv(p, index=False)
        elif p.suffix in [".xlsx", ".xls"]:
            try:
                df.to_excel(p, index=False)
            except ImportError as e:
                raise RuntimeError("Excel support requires openpyxl or xlwt") from e
        else:
            raise ValueError(f"Unsupported file type for saving: {p.suffix}")
        logger.info(f"Saved DataFrame to {p}")
        return p
    except Exception as e:
        logger.error(f"Failed to save DataFrame to {p}: {e}")
        raise RuntimeError(f"Failed to save DataFrame to {p}: {e}") from e


def load_dataframe(path: Union[str, Path]) -> pd.DataFrame:
    """Load DataFrame from Parquet, CSV, or Excel."""
    p = Path(path)
    try:
        if p.suffix == ".parquet":
            df = pd.read_parquet(p)
        elif p.suffix == ".csv":
            df = pd.read_csv(p)
        elif p.suffix in [".xlsx", ".xls"]:
            df = pd.read_excel(p)
        else:
            raise ValueError(f"Unsupported file type for loading: {p.suffix}")
        logger.info(f"Loaded DataFrame from {p} (rows={len(df)})")
        return df
    except Exception as e:
        logger.error(f"Failed to load DataFrame from {p}: {e}")
        raise RuntimeError(f"Failed to load DataFrame from {p}: {e}") from e


def save_json(obj: Any, path: Union[str, Path]) -> Path:
    """Save JSON-serializable object to file. Returns path."""
    p = Path(path)
    ensure_dir(p.parent)
    try:
        with p.open("w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
        logger.info(f"Saved JSON to {p}")
        return p
    except Exception as e:
        logger.error(f"Failed to save JSON to {p}: {e}")
        raise RuntimeError(f"Failed to save JSON to {p}: {e}") from e


def load_json(path: Union[str, Path]) -> Any:
    """Load JSON object (dict, list, etc.) from file."""
    p = Path(path)
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Loaded JSON from {p}")
        return data
    except Exception as e:
        logger.error(f"Failed to load JSON from {p}: {e}")
        raise RuntimeError(f"Failed to load JSON from {p}: {e}") from e


def safe_load(path: Union[str, Path]) -> Any:
    """Convenience loader: detects file type and loads appropriately."""
    p = Path(path)
    if p.suffix in [".parquet", ".csv", ".xlsx", ".xls"]:
        return load_dataframe(p)
    elif p.suffix == ".json":
        return load_json(p)
    else:
        raise ValueError(f"Unsupported file type for safe_load: {p.suffix}")


def safe_save(obj: Any, path: Union[str, Path]) -> Path:
    """Convenience saver: detects object type and saves appropriately."""
    p = Path(path)
    if isinstance(obj, pd.DataFrame):
        return save_dataframe(obj, p)
    else:
        return save_json(obj, p)


def safe_copy(src: Union[str, Path], dest: Union[str, Path]) -> Path:
    """Copy a file from src to dest with logging."""
    src_path = Path(src)
    dest_path = Path(dest)
    ensure_dir(dest_path.parent)
    try:
        shutil.copy2(src_path, dest_path)
        logger.info(f"Copied file from {src_path} to {dest_path}")
        return dest_path
    except Exception as e:
        logger.error(f"Failed to copy file from {src_path} to {dest_path}: {e}")
        raise RuntimeError(f"Failed to copy file from {src_path} to {dest_path}: {e}") from e


def safe_move(src: Union[str, Path], dest: Union[str, Path]) -> Path:
    """Move a file from src to dest with logging."""
    src_path = Path(src)
    dest_path = Path(dest)
    ensure_dir(dest_path.parent)
    try:
        shutil.move(str(src_path), str(dest_path))
        logger.info(f"Moved file from {src_path} to {dest_path}")
        return dest_path
    except Exception as e:
        logger.error(f"Failed to move file from {src_path} to {dest_path}: {e}")
        raise RuntimeError(f"Failed to move file from {src_path} to {dest_path}: {e}") from e


def safe_delete(path: Union[str, Path]) -> None:
    """Delete a file safely with logging."""
    p = Path(path)
    try:
        if p.exists():
            p.unlink()
            logger.info(f"Deleted file: {p}")
        else:
            logger.warning(f"File not found for deletion: {p}")
    except Exception as e:
        logger.error(f"Failed to delete file {p}: {e}")
        raise RuntimeError(f"Failed to delete file {p}: {e}") from e


def safe_exists(path: Union[str, Path]) -> bool:
    """Check if a file exists and log the result."""
    p = Path(path)
    exists = p.exists()
    if exists:
        logger.info(f"File exists: {p}")
    else:
        logger.warning(f"File does not exist: {p}")
    return exists
