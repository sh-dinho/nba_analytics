# ============================================================
# File: core/data_cache.py
# Purpose: Factory for safe caching utilities with validation and archiving
# ============================================================

import pandas as pd
import datetime
import shutil
from pathlib import Path

from core.paths import DATA_DIR, ARCHIVE_DIR
from core.log_config import init_global_logger
from core.exceptions import FileError

logger = init_global_logger()

def make_cache(file_name: str, expected_columns: set[str]) -> dict:
    cache_file = Path(DATA_DIR) / file_name

    def validate(df: pd.DataFrame) -> bool:
        missing = expected_columns - set(df.columns)
        if missing:
            logger.error(f"❌ Cache {file_name} missing columns: {missing}")
            return False
        return True

    def archive():
        if cache_file.exists():
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_file = ARCHIVE_DIR / f"{file_name.replace('.csv','')}_{ts}.csv"
            ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy(cache_file, archive_file)
            logger.info(f"📦 Archived {file_name} to {archive_file}")

    def load() -> pd.DataFrame:
        if cache_file.exists():
            try:
                df = pd.read_csv(cache_file)
                logger.info(f"✅ Loaded cache {file_name} ({len(df)} rows)")
                if not validate(df):
                    raise FileError(f"Cache validation failed: {file_name}", file_path=str(cache_file))
                return df
            except Exception as e:
                raise FileError(f"Failed to read cache: {file_name}", file_path=str(cache_file)) from e
        logger.warning(f"⚠️ Cache {file_name} not found, returning empty DataFrame")
        return pd.DataFrame(columns=list(expected_columns))

    def save(df: pd.DataFrame):
        if not validate(df):
            raise FileError(f"Invalid DataFrame schema for {file_name}", file_path=str(cache_file))
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            archive()
            df.to_csv(cache_file, index=False)
            logger.info(f"💾 Saved cache {file_name} ({len(df)} rows)")
        except Exception as e:
            raise FileError(f"Failed to save cache: {file_name}", file_path=str(cache_file)) from e

    return {"load": load, "save": save, "archive": archive}
