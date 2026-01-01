from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Ingestion Pipeline
# File: src/ingestion/pipeline.py
# Author: Sadiq
#
# Description:
#     Primary ingestion entrypoint.
#
#     Responsibilities:
#       • fetch ScoreboardV3
#       • normalize → wide → long
#       • canonicalize
#       • apply fallbacks
#       • validate
#       • update LONG_SNAPSHOT
#
#     Returns canonical long-format team-game rows.
# ============================================================

import os
from datetime import date
from typing import Iterable
import pandas as pd
from loguru import logger

from src.config.paths import LONG_SNAPSHOT
from src.ingestion.collector import fetch_scoreboard_for_date
from src.ingestion.normalizer.scoreboard_normalizer import normalize_scoreboard_to_wide
from src.ingestion.normalizer.wide_to_long import wide_to_long
from src.ingestion.normalizer.canonicalizer import canonicalize_team_game_df
from src.ingestion.validator.team_game_validator import validate_team_game_df
from src.ingestion.fallback.manager import FallbackManager
from src.ingestion.fallback.schedule_fallback import SeasonScheduleFallback


def _dedupe_and_sort(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return (
        df.drop_duplicates(subset=["game_id", "team"])
        .sort_values(["date", "game_id", "team"])
        .reset_index(drop=True)
    )


def _update_snapshot_atomically(new_rows: pd.DataFrame):
    """Writes to a temporary file before replacing the target to prevent corruption."""
    if LONG_SNAPSHOT.exists():
        existing = pd.read_parquet(LONG_SNAPSHOT)
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        combined = new_rows

    combined = _dedupe_and_sort(combined)

    temp_path = LONG_SNAPSHOT.with_suffix(".tmp.parquet")
    combined.to_parquet(temp_path, index=False)
    os.replace(temp_path, LONG_SNAPSHOT)

    logger.success(f"[Ingestion] Updated {LONG_SNAPSHOT.name} (Total rows: {len(combined)})")


def _process_date_to_memory(day: date) -> pd.DataFrame:
    """Shared logic to fetch and validate a date without persisting to disk."""
    df_raw = fetch_scoreboard_for_date(day)
    wide = normalize_scoreboard_to_wide(df_raw)
    long = wide_to_long(wide)
    long = canonicalize_team_game_df(long)

    if long.empty:
        return long

    # Apply fallbacks
    fb_manager = FallbackManager([SeasonScheduleFallback()])
    long = fb_manager.fill_missing_for_date(day, long)

    # Validate
    validate_team_game_df(long, raise_on_error=True)
    return long


def ingest_dates(dates: Iterable[date]) -> pd.DataFrame:
    """Batch ingestion: Collects all data in memory before a single write."""
    all_new_data = []

    for d in dates:
        try:
            df_day = _process_date_to_memory(d)
            if not df_day.empty:
                all_new_data.append(df_day)
        except Exception as e:
            logger.error(f"Failed to process {d}: {e}")

    if not all_new_data:
        return pd.DataFrame()

    full_batch = pd.concat(all_new_data, ignore_index=True)
    _update_snapshot_atomically(full_batch)

    return full_batch


def ingest_single_date(day: date) -> pd.DataFrame:
    """Convenience wrapper."""
    return ingest_dates([day])