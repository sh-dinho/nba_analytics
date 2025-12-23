from __future__ import annotations
import pandas as pd
from datetime import date, timedelta
from loguru import logger

from src.ingestion.pipeline import ingest_dates
from src.config.paths import SCHEDULE_SNAPSHOT


def find_missing_dates() -> list[date]:
    """
    Identify dates between min(date) and max(date) that have zero games.
    """
    if not SCHEDULE_SNAPSHOT.exists():
        logger.warning("No schedule snapshot found.")
        return []

    df = pd.read_parquet(SCHEDULE_SNAPSHOT)
    df["date"] = pd.to_datetime(df["date"]).dt.date

    min_d = df["date"].min()
    max_d = df["date"].max()

    all_days = [min_d + timedelta(days=i) for i in range((max_d - min_d).days + 1)]
    existing_days = set(df["date"])

    missing = [d for d in all_days if d not in existing_days]
    logger.info(f"Found {len(missing)} missing dates.")
    return missing


def reingest_missing_dates(workers: int = 8):
    """
    Automatically re-ingest all missing dates.
    """
    missing = find_missing_dates()
    if not missing:
        logger.success("No missing dates detected. Ingestion is complete.")
        return

    logger.info(f"Re-ingesting {len(missing)} missing dates...")
    ingest_dates(missing, use_cache=False, force=True, workers=workers)
    logger.success("Re-ingestion of missing dates complete.")
