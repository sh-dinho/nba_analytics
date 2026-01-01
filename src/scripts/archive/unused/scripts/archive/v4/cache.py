from __future__ import annotations

import pandas as pd
from pathlib import Path
from loguru import logger
from datetime import date

# Central cache directory (matches ingestion pipeline)
CACHE_DIR = Path("data/cache/scoreboard_v3")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Path helper
# ------------------------------------------------------------


def _cache_path(target_date: date) -> Path:
    """Return the cache file path for a given date."""
    return CACHE_DIR / f"{target_date.strftime('%Y-%m-%d')}.parquet"


# ------------------------------------------------------------
# Load from cache
# ------------------------------------------------------------


def load_from_cache(target_date: date) -> pd.DataFrame | None:
    """
    Load cached scoreboard data for a given date.
    Returns:
        DataFrame if cache exists and is valid
        None if no cache or cache is corrupted
    """
    path = _cache_path(target_date)

    if not path.exists():
        logger.debug(f"Cache miss for {target_date}")
        return None

    try:
        df = pd.read_parquet(path)
        logger.debug(f"Cache hit for {target_date} → {path}")
        return df
    except Exception as e:
        logger.warning(f"Cache corrupted for {target_date}: {e}")
        return None


# ------------------------------------------------------------
# Save to cache
# ------------------------------------------------------------


def save_to_cache(df: pd.DataFrame, target_date: date) -> None:
    """
    Save scoreboard data to cache.
    Does NOT cache empty DataFrames.
    """
    if df is None or df.empty:
        logger.debug(f"No data to cache for {target_date}")
        return

    path = _cache_path(target_date)

    try:
        df.to_parquet(path, index=False)
        logger.debug(f"Saved cache for {target_date} → {path}")
    except Exception as e:
        logger.error(f"Failed to save cache for {target_date}: {e}")
