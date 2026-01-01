from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Ingestion Orchestrator
# File: src/ingestion/orchestrator.py
# Author: Sadiq
#
# Description:
#     Central ingestion orchestrator. Handles:
#       • fetch ScoreboardV3
#       • normalize → wide → long
#       • canonicalize
#       • apply fallbacks
#       • validate
#       • return canonical long-format rows
#
#     This module does NOT:
#       • build features
#       • manage snapshots
#       • handle versioning
#
#     It is a pure orchestrator for ingestion logic.
# ============================================================

from datetime import date
from typing import Iterable, List

import pandas as pd
from loguru import logger

from src.ingestion.collector import fetch_scoreboard_for_date
from src.ingestion.normalizer.scoreboard_normalizer import normalize_scoreboard_to_wide
from src.ingestion.normalizer.wide_to_long import wide_to_long
from src.ingestion.normalizer.canonicalizer import canonicalize_team_game_df
from src.ingestion.validator.team_game_validator import validate_team_game_df
from src.ingestion.fallback.manager import FallbackManager
from src.ingestion.fallback.schedule_fallback import SeasonScheduleFallback


# ------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------

def _process_single_date(day: date) -> pd.DataFrame:
    """
    Fetch, normalize, canonicalize, fallback, validate.
    Returns canonical long-format rows for a single date.
    """
    logger.info(f"[Orchestrator] Processing {day}")

    # Fetch raw scoreboard
    df_raw = fetch_scoreboard_for_date(day)

    # Normalize → wide → long
    wide = normalize_scoreboard_to_wide(df_raw)
    long = wide_to_long(wide)

    # Canonicalize
    long = canonicalize_team_game_df(long)

    if long.empty:
        return long

    # Apply fallbacks
    fb = FallbackManager([SeasonScheduleFallback()])
    long = fb.fill_missing_for_date(day, long)

    # Validate
    validate_team_game_df(long, raise_on_error=True)

    return long


# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------

def ingest_single_date(day: date) -> pd.DataFrame:
    """
    Ingest a single date and return canonical long-format rows.
    """
    return _process_single_date(day)


def ingest_dates(dates: Iterable[date]) -> pd.DataFrame:
    """
    Ingest multiple dates and return canonical long-format rows.
    """
    dates = list(dates)
    if not dates:
        logger.warning("[Orchestrator] ingest_dates called with no dates.")
        return pd.DataFrame()

    logger.info(
        f"[Orchestrator] Ingesting {len(dates)} dates "
        f"(start={dates[0]}, end={dates[-1]})"
    )

    all_rows: List[pd.DataFrame] = []

    for d in dates:
        try:
            df_day = _process_single_date(d)
            if not df_day.empty:
                all_rows.append(df_day)
        except Exception as e:
            logger.error(f"[Orchestrator] Failed to ingest {d}: {e}")

    if not all_rows:
        return pd.DataFrame()

    return pd.concat(all_rows, ignore_index=True)