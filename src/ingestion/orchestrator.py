from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Ingestion Orchestrator
# File: src/ingestion/orchestrator.py
# Author: Sadiq
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


# Instantiate fallbacks once
FALLBACKS = FallbackManager([SeasonScheduleFallback()])


# ------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------

def _process_single_date(day: date) -> pd.DataFrame:
    """
    Fetch, normalize, canonicalize, fallback, validate.
    Returns canonical long-format rows for a single date.
    """
    logger.info(f"[Orchestrator] Processing {day}")

    # --------------------------------------------------------
    # Fetch raw scoreboard
    # --------------------------------------------------------
    df_raw = fetch_scoreboard_for_date(day)
    logger.debug(f"[Orchestrator] Raw rows: {len(df_raw)}")

    # --------------------------------------------------------
    # Normalize → wide → long
    # --------------------------------------------------------
    try:
        wide = normalize_scoreboard_to_wide(df_raw)
        logger.debug(f"[Orchestrator] Wide rows: {len(wide)}")

        long = wide_to_long(wide)
        logger.debug(f"[Orchestrator] Long rows: {len(long)}")
    except Exception as e:
        logger.error(f"[Orchestrator] Normalization failed for {day}: {e}")
        return pd.DataFrame()

    # --------------------------------------------------------
    # Canonicalize
    # --------------------------------------------------------
    try:
        long = canonicalize_team_game_df(long)
        logger.debug(f"[Orchestrator] Canonical rows: {len(long)}")
    except Exception as e:
        logger.error(f"[Orchestrator] Canonicalization failed for {day}: {e}")
        return pd.DataFrame()

    if long.empty:
        logger.warning(f"[Orchestrator] No games found for {day}")
        return long

    # --------------------------------------------------------
    # Apply fallbacks
    # --------------------------------------------------------
    try:
        long = FALLBACKS.fill_missing_for_date(day, long)
    except Exception as e:
        logger.error(f"[Orchestrator] Fallback failed for {day}: {e}")
        return pd.DataFrame()

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------
    validate_team_game_df(long, raise_on_error=True)

    return long


# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------

def ingest_single_date(day: date) -> pd.DataFrame:
    """Ingest a single date and return canonical long-format rows."""
    return _process_single_date(day)


def ingest_dates(dates: Iterable[date]) -> pd.DataFrame:
    """Ingest multiple dates and return canonical long-format rows."""
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


# ------------------------------------------------------------
# Canonical Entry Point
# ------------------------------------------------------------

def run_full_ingestion() -> pd.DataFrame:
    """Canonical ingestion entrypoint for today's games."""
    today = date.today()
    logger.info(f"[Orchestrator] Running full ingestion for {today}")

    df = ingest_single_date(today)

    if df.empty:
        logger.warning("[Orchestrator] No games ingested for today.")
    else:
        logger.success(f"[Orchestrator] Ingested {len(df)} rows for {today}")

    return df
