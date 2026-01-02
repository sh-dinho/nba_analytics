from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Ingestion Pipeline
# File: src/ingestion/pipeline.py
# Author: Sadiq
# ============================================================

import os
from datetime import date
from typing import Iterable, List

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


# Instantiate fallbacks once
FALLBACKS = FallbackManager([SeasonScheduleFallback()])


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _dedupe_and_sort(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return (
        df.drop_duplicates(subset=["game_id", "team"])
        .sort_values(["date", "game_id", "team"])
        .reset_index(drop=True)
    )


def _update_snapshot_atomically(new_rows: pd.DataFrame) -> None:
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

    # Verification
    if not LONG_SNAPSHOT.exists():
        raise FileNotFoundError(f"Verification Failed: {LONG_SNAPSHOT} was not created.")

    verification_df = pd.read_parquet(LONG_SNAPSHOT)
    if verification_df.empty:
        raise ValueError(f"Verification Failed: {LONG_SNAPSHOT} is empty after write.")

    logger.success(
        f"[Ingestion] Snapshot updated → {LONG_SNAPSHOT.name} "
        f"(Total rows: {len(combined)}, New rows: {len(new_rows)})"
    )


def _process_date_to_memory(day: date) -> pd.DataFrame:
    """Fetch, normalize, canonicalize, fallback, validate — all in memory."""
    logger.info(f"[Ingestion] Processing {day}")

    # Fetch
    df_raw = fetch_scoreboard_for_date(day)
    logger.debug(f"[Ingestion] Raw rows: {len(df_raw)}")

    # Normalize → wide → long
    try:
        wide = normalize_scoreboard_to_wide(df_raw)
        logger.debug(f"[Ingestion] Wide rows: {len(wide)}")

        long = wide_to_long(wide)
        logger.debug(f"[Ingestion] Long rows: {len(long)}")
    except Exception as e:
        logger.error(f"[Ingestion] Normalization failed for {day}: {e}")
        return pd.DataFrame()

    # Canonicalize
    try:
        long = canonicalize_team_game_df(long)
        logger.debug(f"[Ingestion] Canonical rows: {len(long)}")
    except Exception as e:
        logger.error(f"[Ingestion] Canonicalization failed for {day}: {e}")
        return pd.DataFrame()

    if long.empty:
        logger.warning(f"[Ingestion] No games found for {day}")
        return long

    # Fallbacks
    try:
        long = FALLBACKS.fill_missing_for_date(day, long)
    except Exception as e:
        logger.error(f"[Ingestion] Fallback failed for {day}: {e}")
        return pd.DataFrame()

    # Validate
    validate_team_game_df(long, raise_on_error=True)

    return long


# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------

def ingest_dates(dates: Iterable[date]) -> pd.DataFrame:
    """Batch ingestion: Collects all data in memory before a single verified write."""
    dates = list(dates)
    if not dates:
        logger.warning("[Ingestion] ingest_dates called with no dates.")
        return pd.DataFrame()

    logger.info(
        f"[Ingestion] Ingesting {len(dates)} dates "
        f"(start={dates[0]}, end={dates[-1]})"
    )

    all_new_data: List[pd.DataFrame] = []

    for d in dates:
        try:
            df_day = _process_date_to_memory(d)
            if not df_day.empty:
                all_new_data.append(df_day)
        except Exception as e:
            logger.error(f"[Ingestion] Failed to process {d}: {e}")

    if not all_new_data:
        logger.warning("[Ingestion] No new rows ingested.")
        return pd.DataFrame()

    full_batch = pd.concat(all_new_data, ignore_index=True)
    _update_snapshot_atomically(full_batch)

    return full_batch


def ingest_single_date(day: date) -> pd.DataFrame:
    """Convenience wrapper."""
    return ingest_dates([day])
