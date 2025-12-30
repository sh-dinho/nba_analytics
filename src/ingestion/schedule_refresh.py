from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v4
# Module: Schedule Refresh (Smart + Lightweight)
# File: src/ingestion/schedule_refresh.py
# ============================================================

from datetime import date, timedelta
import pandas as pd
from loguru import logger

from src.config.paths import SCHEDULE_SNAPSHOT
from src.ingestion.pipeline import ingest_dates
from src.ingestion.schedule_scraper import build_season_schedule


def refresh_schedule_if_needed(today: date) -> pd.DataFrame:
    """
    v4 logic:
    - If schedule is missing → rebuild full season
    - If schedule exists but is behind → ingest only missing days
    - Never rebuild full season during daily predictions
    """

    # Case 1 — No schedule exists at all
    if not SCHEDULE_SNAPSHOT.exists():
        logger.warning("[ScheduleRefresh] No schedule found — full rebuild required.")
        season_start_year = today.year if today.month >= 7 else today.year - 1
        return build_season_schedule(season_start_year)

    # Case 2 — Schedule exists
    df = pd.read_parquet(SCHEDULE_SNAPSHOT)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    last_date = df["date"].max()

    # Already fresh
    if last_date >= today:
        logger.info("[ScheduleRefresh] Schedule is fresh.")
        return df

    # Case 3 — Only ingest missing days
    missing_days = pd.date_range(last_date + timedelta(days=1), today)
    missing_days = [d.date() for d in missing_days]

    logger.info(f"[ScheduleRefresh] Ingesting missing days: {missing_days}")

    return ingest_dates(missing_days, feature_version="v1")
