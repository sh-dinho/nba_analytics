from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v4
# Module: Missing Dates Re-ingestion
# Author: Sadiq
# ============================================================

import pandas as pd
from datetime import date, timedelta
from loguru import logger

from src.config.paths import SCHEDULE_SNAPSHOT
from src.ingestion.pipeline import ingest_dates


def find_missing_dates() -> list[date]:
    """
    Identify dates where games SHOULD exist but ingestion is missing.
    Rules:
      - Only dates between min(date) and max(date)
      - Only dates where NBA actually had games historically
      - Avoids re-ingesting normal off-days
    """
    if not SCHEDULE_SNAPSHOT.exists():
        logger.warning("No schedule snapshot found.")
        return []

    df = pd.read_parquet(SCHEDULE_SNAPSHOT)
    df["date"] = pd.to_datetime(df["date"]).dt.date

    # Count games per day
    counts = df.groupby("date").size()

    min_d = counts.index.min()
    max_d = counts.index.max()

    all_days = [min_d + timedelta(days=i) for i in range((max_d - min_d).days + 1)]

    missing = []
    for d in all_days:
        if d not in counts.index:
            # Only consider missing if surrounded by real game days
            prev_day = d - timedelta(days=1)
            next_day = d + timedelta(days=1)

            if prev_day in counts.index or next_day in counts.index:
                missing.append(d)

    logger.info(f"Detected {len(missing)} missing game days.")
    return missing


def reingest_missing_dates(max_workers: int = 8):
    """
    Re-ingest only missing game days.
    """
    missing = find_missing_dates()
    if not missing:
        logger.success("No missing game days detected. Ingestion is complete.")
        return

    logger.info(f"Re-ingesting {len(missing)} missing game days...")
    ingest_dates(
        missing,
        use_cache=False,
        force=True,
        max_workers=max_workers,
        feature_version="v1",
    )
    logger.success("Re-ingestion of missing game days complete.")
