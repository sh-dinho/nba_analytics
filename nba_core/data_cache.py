# ============================================================
# File: core/data_cache.py
# Purpose: Class-based safe caching utilities with validation and archiving
# ============================================================

import pandas as pd
import datetime
import shutil
from pathlib import Path

from nba_core.paths import DATA_DIR, ARCHIVE_DIR
from nba_core.log_config import init_global_logger
from nba_core.exceptions import FileError

logger = init_global_logger()


class Cache:
    """
    Cache provides safe load/save/archive utilities for CSV-based data caches.
    - Validates schema against expected columns
    - Archives old versions before overwriting
    - Ensures atomic saves to prevent corruption
    """

    def __init__(self, file_name: str, expected_columns: set[str]):
        self.file_name = file_name
        self.cache_file = Path(DATA_DIR) / file_name
        self.expected_columns = expected_columns

    # -------------------------------
    # Validation
    # -------------------------------
    def validate(self, df: pd.DataFrame) -> bool:
        """Ensure DataFrame contains all expected columns."""
        missing = self.expected_columns - set(df.columns)
        if missing:
            logger.error(f"❌ Cache {self.file_name} missing columns: {missing}")
            return False
        return True

    # -------------------------------
    # Archiving
    # -------------------------------
    def archive(self):
        """Archive current cache file with timestamp suffix."""
        if self.cache_file.exists():
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_file = ARCHIVE_DIR / f"{self.file_name.replace('.csv','')}_{ts}.csv"
            ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy(self.cache_file, archive_file)
            logger.info(f"📦 Archived {self.file_name} to {archive_file}")

    # -------------------------------
    # Load
    # -------------------------------
    def load(self) -> pd.DataFrame:
        """Load cache file into DataFrame, validating schema."""
        if self.cache_file.exists():
            try:
                df = pd.read_csv(self.cache_file)
                logger.info(f"✅ Loaded cache {self.file_name} ({len(df)} rows)")
                if not self.validate(df):
                    raise FileError(f"Cache validation failed: {self.file_name}", file_path=str(self.cache_file))
                return df
            except Exception as e:
                raise FileError(f"Failed to read cache: {self.file_name}", file_path=str(self.cache_file)) from e
        logger.warning(f"⚠️ Cache {self.file_name} not found, returning empty DataFrame")
        return pd.DataFrame(columns=list(self.expected_columns))

    # -------------------------------
    # Save
    # -------------------------------
    def save(self, df: pd.DataFrame):
        """Save DataFrame to cache file with validation and atomic write."""
        if not self.validate(df):
            raise FileError(f"Invalid DataFrame schema for {self.file_name}", file_path=str(self.cache_file))
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            self.archive()
            tmp_file = self.cache_file.with_suffix(".tmp")
            df.to_csv(tmp_file, index=False)
            tmp_file.replace(self.cache_file)
            logger.info(f"💾 Saved cache {self.file_name} ({len(df)} rows)")
        except Exception as e:
            raise FileError(f"Failed to save cache: {self.file_name}", file_path=str(self.cache_file)) from e
