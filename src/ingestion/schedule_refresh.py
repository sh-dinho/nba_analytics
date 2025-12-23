from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v4
# Module: Schedule Refresh
# File: src/ingestion/schedule_refresh.py
#
# Description:
#     Automatically ensures that season_schedule.parquet exists
#     and is reasonably fresh. If missing or stale, it triggers
#     a re-scrape from ESPN.
#
#     Public API:
#       - refresh_schedule_if_needed()
# ============================================================

from datetime import datetime, timedelta
from pathlib import Path

from loguru import logger

from src.config.paths import SEASON_SCHEDULE_PATH
from src.ingestion.schedule_scraper import build_season_schedule


def _guess_current_season_start_year(today: datetime) -> int:
    """
    For a given date, infer the season_start_year.
    Example:
      - Nov 2024 → 2024 (2024-25 season)
      - Mar 2025 → 2024 (still 2024-25)
      - Jul 2025 → 2025 (next season if scraping early)
    """
    year = today.year
    if today.month >= 7:
        return year
    return year - 1


def _is_stale(path: Path, max_age_days: int = 1) -> bool:
    if not path.exists():
        return True
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    return (datetime.now() - mtime) > timedelta(days=max_age_days)


def refresh_schedule_if_needed(max_age_days: int = 1) -> None:
    """
    Ensure SEASON_SCHEDULE_PATH exists and is not older than
    max_age_days. If missing or stale, re-scrape from ESPN.
    """
    if not _is_stale(SEASON_SCHEDULE_PATH, max_age_days=max_age_days):
        logger.info(
            f"[ScheduleRefresh] Season schedule at {SEASON_SCHEDULE_PATH} "
            f"is fresh (<= {max_age_days} days old)."
        )
        return

    today = datetime.now()
    season_start_year = _guess_current_season_start_year(today)
    logger.warning(
        f"[ScheduleRefresh] Season schedule missing or stale. "
        f"Rebuilding for season {season_start_year}-{season_start_year + 1}."
    )
    build_season_schedule(season_start_year=season_start_year)
