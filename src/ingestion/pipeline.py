from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v4
# Module: Ingestion Pipeline (Smart + Cached + Parallel)
# File: src/ingestion/pipeline.py
# Author: Sadiq
# ============================================================

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, List

import pandas as pd
from loguru import logger
from requests.exceptions import RequestException

from src.config.paths import (
    DATA_DIR,
    FEATURES_DIR,
    LONG_SNAPSHOT,
    SCHEDULE_SNAPSHOT,
)
from src.features.builder import FeatureBuilder
from src.ingestion import normalizer, validator, wide_to_long
from src.ingestion.collector import fetch_scoreboard

# Cache directory for raw scoreboard data
CACHE_DIR = DATA_DIR / "cache" / "scoreboard_v3"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------
# Cache helpers
# -----------------------------------------------------------


def _cache_path(target_date: date) -> Path:
    return CACHE_DIR / f"{target_date.strftime('%Y-%m-%d')}.parquet"


def _load_from_cache(target_date: date) -> pd.DataFrame | None:
    path = _cache_path(target_date)
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        logger.debug(f"Cache hit for {target_date} → {path}")
        return df
    except Exception as e:
        logger.warning(f"Failed to read cache for {target_date}: {e}")
        return None


def _save_to_cache(target_date: date, df: pd.DataFrame) -> None:
    path = _cache_path(target_date)
    try:
        df.to_parquet(path, index=False)
        logger.debug(f"Cached scoreboard for {target_date} → {path}")
    except Exception as e:
        logger.warning(f"Failed to write cache for {target_date}: {e}")


# -----------------------------------------------------------
# Per-date ingestion with retry and enhanced error handling
# -----------------------------------------------------------


def fetch_scoreboard_with_retry(
    target_date: date, retries: int = 3, delay: int = 2
) -> pd.DataFrame:
    """
    Fetch scoreboard data with retry mechanism in case of failures.
    """
    attempt = 0
    while attempt < retries:
        try:
            return fetch_scoreboard(target_date)
        except RequestException as e:
            logger.warning(f"Error fetching scoreboard for {target_date}: {e}")
            attempt += 1
            time.sleep(delay)
        except Exception as e:
            logger.error(f"Unexpected error fetching {target_date}: {e}")
            break
    return pd.DataFrame()  # Return empty DataFrame on failure


def process_date(
    target_date: date, use_cache: bool = True, force: bool = False
) -> pd.DataFrame:
    """
    Fetch, normalize, validate, and return canonical schedule rows for a single date.
    """
    is_future_or_today = target_date >= date.today()

    # Try cache first
    if use_cache and not force and not is_future_or_today:
        cached = _load_from_cache(target_date)
        if cached is not None:
            return cached

    # Fetch raw scoreboard with retries
    raw_df = fetch_scoreboard_with_retry(target_date)
    if raw_df.empty:
        if not is_future_or_today:
            _save_to_cache(target_date, pd.DataFrame())
        logger.debug(f"No games for {target_date}.")
        return pd.DataFrame()

    # Normalize
    wide_or_long_df = normalizer.normalize_schedule(raw_df)
    if wide_or_long_df.empty:
        if not is_future_or_today:
            _save_to_cache(target_date, pd.DataFrame())
        logger.debug(f"Normalizer produced empty DataFrame for {target_date}.")
        return pd.DataFrame()

    # At this point, normalizer already returns TEAM-GAME rows (long format v4),
    # but validator expects canonical team-game schema, so we just validate.
    validated = validator.validate_ingestion_dataframe(wide_or_long_df)

    # Cache validated result
    if not is_future_or_today:
        _save_to_cache(target_date, validated)

    return validated


# -----------------------------------------------------------
# Snapshot update helpers with Long Snapshot Bootstrap
# -----------------------------------------------------------


def bootstrap_long_snapshot():
    """
    If the long snapshot is missing, attempt to rebuild it from the schedule snapshot.
    """
    if not LONG_SNAPSHOT.exists():
        if SCHEDULE_SNAPSHOT.exists():
            logger.warning(
                "⚠️ Long snapshot missing — rebuilding from schedule snapshot"
            )
            schedule_df = pd.read_parquet(SCHEDULE_SNAPSHOT)
            canonical_long_df = wide_to_long.wide_to_long(schedule_df)
            LONG_SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
            canonical_long_df.to_parquet(LONG_SNAPSHOT, index=False)
            logger.success("Long snapshot created from schedule.")
        else:
            logger.error("❌ No schedule data available to bootstrap.")
            return


def _merge_into_schedule(new_data: pd.DataFrame) -> pd.DataFrame:
    """
    Merge new canonical rows into the master schedule snapshot.
    """
    if new_data.empty:
        logger.warning("No new data provided to _merge_into_schedule.")
        if SCHEDULE_SNAPSHOT.exists():
            return pd.read_parquet(SCHEDULE_SNAPSHOT)
        return pd.DataFrame()

    if SCHEDULE_SNAPSHOT.exists():
        existing = pd.read_parquet(SCHEDULE_SNAPSHOT)
        existing["date"] = pd.to_datetime(existing["date"]).dt.date
        merged = pd.concat([existing, new_data], ignore_index=True)
        merged = merged.drop_duplicates(subset=["game_id", "team"], keep="last")
    else:
        merged = new_data.copy()

    merged["date"] = pd.to_datetime(merged["date"]).dt.date
    merged.sort_values("date", inplace=True)
    SCHEDULE_SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(SCHEDULE_SNAPSHOT, index=False)

    logger.success(
        f"Schedule snapshot updated → {SCHEDULE_SNAPSHOT} (rows={len(merged)})"
    )
    return merged


def _update_long_and_features(
    schedule_df: pd.DataFrame, feature_version: str = "v1"
) -> None:
    """
    Build long-format and feature tables from the canonical schedule.
    """
    if schedule_df.empty:
        logger.warning("Schedule is empty; skipping long/feature update.")
        return

    # Wide → Long (fallback) or pass-through if already long
    long_df = wide_to_long.wide_to_long(schedule_df)
    LONG_SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_parquet(LONG_SNAPSHOT, index=False)
    logger.info(f"Long-format snapshot updated → {LONG_SNAPSHOT} (rows={len(long_df)})")

    # Long → Features (v1 feature set, used as ingestion artifact)
    fb = FeatureBuilder(version=feature_version)
    features_df = fb.build_from_long(long_df)

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    feature_path = FEATURES_DIR / f"features_{feature_version}.parquet"
    features_df.to_parquet(feature_path, index=False)

    logger.success(
        f"Features snapshot (version={feature_version}) updated → {feature_path} "
        f"(rows={len(features_df)}, cols={len(features_df.columns)})"
    )


# -----------------------------------------------------------
# Parallel ingestion with retry and enhanced error handling
# -----------------------------------------------------------


def ingest_dates(
    dates: Iterable[date],
    use_cache: bool = True,
    force: bool = False,
    max_workers: int = 8,
    feature_version: str = "v1",
) -> pd.DataFrame:
    """
    Ingest an arbitrary collection of dates in parallel.
    """
    dates_list = list(dates)
    if not dates_list:
        logger.warning("No dates provided to ingest_dates.")
        if SCHEDULE_SNAPSHOT.exists():
            return pd.read_parquet(SCHEDULE_SNAPSHOT)
        return pd.DataFrame()

    logger.info(
        f"🏀 Ingesting {len(dates_list)} dates (parallel, max_workers={max_workers})"
    )
    results: List[pd.DataFrame] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_date, d, use_cache, force): d for d in dates_list
        }
        for future in as_completed(futures):
            d = futures[future]
            try:
                df = future.result()
            except Exception as e:
                logger.warning(f"Error processing {d}: {e}")
                continue

            if df.empty:
                logger.debug(f"No games ingested for {d}.")
            else:
                results.append(df)
                logger.info(f"Ingested {len(df)} rows for {d}.")

    # --- PATCHED LOGIC STARTS HERE ---
    if not results:
        logger.warning("No new games ingested for provided dates.")
        if SCHEDULE_SNAPSHOT.exists():
            schedule_df = pd.read_parquet(SCHEDULE_SNAPSHOT)

            # If long snapshot is missing or empty, rebuild it from schedule
            needs_long_refresh = (
                not LONG_SNAPSHOT.exists() or pd.read_parquet(LONG_SNAPSHOT).empty
            )

            if needs_long_refresh and not schedule_df.empty:
                logger.warning(
                    "LONG_SNAPSHOT missing or empty but schedule exists — "
                    "rebuilding long + features from schedule."
                )
                _update_long_and_features(schedule_df, feature_version=feature_version)

            return schedule_df

        return pd.DataFrame()
    # --- PATCHED LOGIC ENDS HERE ---

    new_df = pd.concat(results, ignore_index=True)
    merged = _merge_into_schedule(new_df)
    _update_long_and_features(merged, feature_version=feature_version)
    return merged


# -----------------------------------------------------------
# Smart season ingestion with enhanced error handling
# -----------------------------------------------------------


def smart_ingest_season(
    season_start_year: int,
    use_cache: bool = True,
    force: bool = False,
    max_empty_days: int = 30,
    feature_version: str = "v1",
) -> pd.DataFrame:
    """
    Ingest an entire NBA season (regular + play-in + playoffs).

    - Starts at Oct 1 of season_start_year
    - Ends at Jul 1 of next year
    - Stops early if we see too many consecutive empty days (safety rail)
    """
    start_date = date(season_start_year, 10, 1)
    end_date = date(season_start_year + 1, 7, 1)

    logger.info(
        f"🏀 Smart season ingestion: {season_start_year}-{season_start_year + 1} "
        f"({start_date} → {end_date})"
    )

    all_dates: List[date] = []
    current = start_date
    while current <= end_date:
        all_dates.append(current)
        current += timedelta(days=1)

    consecutive_empty = 0
    results: List[pd.DataFrame] = []

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(process_date, d, use_cache, force): d for d in all_dates
        }
        for future in as_completed(futures):
            d = futures[future]
            try:
                df = future.result()
            except Exception as e:
                logger.warning(f"Error processing {d} during season ingest: {e}")
                continue

            if df.empty:
                consecutive_empty += 1
                if consecutive_empty % 10 == 0:
                    logger.debug(
                        f"{consecutive_empty} consecutive empty days "
                        f"ending at {d} during season ingest."
                    )
                if consecutive_empty >= max_empty_days:
                    logger.warning(
                        f"Reached max_empty_days={max_empty_days} during season ingest "
                        f"(last date={d}). Stopping early."
                    )
                    break
            else:
                consecutive_empty = 0
                results.append(df)
                logger.info(f"[Season] Ingested {len(df)} rows for {d}.")

    if not results:
        logger.warning("Season ingest produced no new games.")
        if SCHEDULE_SNAPSHOT.exists():
            schedule_df = pd.read_parquet(SCHEDULE_SNAPSHOT)
            # Ensure long/features exist if schedule is non-empty
            needs_long_refresh = (
                not LONG_SNAPSHOT.exists() or pd.read_parquet(LONG_SNAPSHOT).empty
            )
            if needs_long_refresh and not schedule_df.empty:
                logger.warning(
                    "LONG_SNAPSHOT missing or empty after season ingest — "
                    "rebuilding long + features from schedule."
                )
                _update_long_and_features(schedule_df, feature_version=feature_version)
            return schedule_df
        return pd.DataFrame()

    new_df = pd.concat(results, ignore_index=True)
    merged = _merge_into_schedule(new_df)
    _update_long_and_features(merged, feature_version=feature_version)
    return merged


# -----------------------------------------------------------
# Daily ingestion (yesterday + today) with improved logging
# -----------------------------------------------------------


def run_today_ingestion(
    feature_version: str = "v1",
    use_cache: bool = True,
    force: bool = False,
    max_workers: int = 2,
) -> pd.DataFrame:
    """
    Ingest yesterday + today, update schedule, long snapshot, and features.

    - Safe on no-game days (returns existing schedule if nothing new)
    - Uses cache for historical dates
    - Designed to be called from the v4 end-to-end runner
    """
    today = date.today()
    yesterday = today - timedelta(days=1)

    target_dates = [yesterday, today]
    logger.info(f"Running today ingestion for {target_dates}")

    # Ensure long snapshot is bootstrapped when possible
    bootstrap_long_snapshot()

    merged_schedule = ingest_dates(
        dates=target_dates,
        use_cache=use_cache,
        force=force,
        max_workers=max_workers,
        feature_version=feature_version,
    )

    if merged_schedule.empty:
        logger.warning(
            "run_today_ingestion: No schedule data after ingest. "
            "This may indicate an API outage or off-days."
        )
    else:
        logger.info(
            f"run_today_ingestion: Schedule rows after ingest = {len(merged_schedule)}"
        )

    return merged_schedule
