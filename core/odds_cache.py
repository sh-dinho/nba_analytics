# ============================================================
# File: core/odds_cache.py
# Purpose: Local caching utility for sportsbook odds data with validation and archiving
# ============================================================

import pandas as pd
import datetime
import shutil
from core.paths import DATA_DIR, ARCHIVE_DIR
from core.log_config import init_global_logger
from core.exceptions import FileError

logger = init_global_logger()

CACHE_FILE = DATA_DIR / "odds_cache.csv"
EXPECTED_COLUMNS = {"team", "odds", "date"}

def validate_odds(df: pd.DataFrame) -> bool:
    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        logger.error(f"❌ Odds cache missing columns: {missing}")
        return False
    return True

def archive_odds():
    if CACHE_FILE.exists():
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_file = ARCHIVE_DIR / f"odds_cache_{ts}.csv"
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy(CACHE_FILE, archive_file)
        logger.info(f"📦 Archived odds cache to {archive_file}")

def load_odds() -> pd.DataFrame:
    if CACHE_FILE.exists():
        try:
            df = pd.read_csv(CACHE_FILE)
            logger.info(f"✅ Loaded odds cache: {CACHE_FILE} ({len(df)} rows)")
            if not validate_odds(df):
                raise FileError("Odds cache validation failed", file_path=str(CACHE_FILE))
            return df
        except Exception as e:
            raise FileError(f"Failed to read odds cache: {CACHE_FILE}", file_path=str(CACHE_FILE)) from e
    logger.warning("⚠️ Odds cache not found, returning empty DataFrame")
    return pd.DataFrame(columns=list(EXPECTED_COLUMNS))

def save_odds(df: pd.DataFrame):
    if not validate_odds(df):
        raise FileError("Invalid odds DataFrame schema", file_path=str(CACHE_FILE))
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        archive_odds()
        df.to_csv(CACHE_FILE, index=False)
        logger.info(f"💾 Saved odds cache: {CACHE_FILE} ({len(df)} rows)")
    except Exception as e:
        raise FileError(f"Failed to save odds cache: {CACHE_FILE}", file_path=str(CACHE_FILE)) from e
