from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Missing Date Detector
# File: src/ingestion/maintenance/missing_dates.py
# Author: Sadiq
#
# Description:
#     Utility for detecting missing dates in the canonical
#     long-format snapshot. Useful for ingestion health checks,
#     backfilling, and drift detection.
# ============================================================

from datetime import date, timedelta
import pandas as pd
from loguru import logger

from src.config.paths import LONG_SNAPSHOT


def detect_missing_dates(start: date, end: date) -> list[date]:
    """
    Return a list of dates between [start, end] that do not appear
    in the long-format snapshot.
    """
    if start > end:
        logger.error("[MissingDates] start > end — invalid range.")
        return []

    if not LONG_SNAPSHOT.exists():
        logger.warning("[MissingDates] LONG_SNAPSHOT does not exist.")
        return []

    # Projection: Only load 'date' column
    df = pd.read_parquet(LONG_SNAPSHOT, columns=["date"])
    df_dates = set(pd.to_datetime(df["date"]).dt.date)

    missing = []
    cur = start
    while cur <= end:
        if cur not in df_dates:
            missing.append(cur)
        cur += timedelta(days=1)

    if missing:
        logger.warning(f"[MissingDates] Missing {len(missing)} dates in range.")
    else:
        logger.success("[MissingDates] No missing dates detected.")

    return missing