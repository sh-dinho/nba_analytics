# ============================================================
# 🏀 NBA Analytics v3
# Module: Ingestion Pipeline
# File: src/ingestion/pipeline.py
# Author: Sadiq
#
# Description:
#     High-level ingestion pipeline that:
#       - fetches scoreboard data for one or more dates,
#       - validates it with strict schema checks,
#       - merges into canonical snapshot,
#       - writes metadata for auditability.
# ============================================================

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd
from loguru import logger

from src.config.paths import DATA_DIR
from src.ingestion.collector import fetch_scoreboard
from src.ingestion.validator import validate_ingestion_dataframe

CANONICAL_SCHEDULE_PATH = DATA_DIR / "canonical" / "schedule.parquet"
CANONICAL_METADATA_PATH = DATA_DIR / "canonical" / "schedule_metadata.json"


def _ensure_dirs():
    (DATA_DIR / "canonical").mkdir(parents=True, exist_ok=True)


def _load_existing_schedule() -> pd.DataFrame:
    if not CANONICAL_SCHEDULE_PATH.exists():
        return pd.DataFrame()
    df = pd.read_parquet(CANONICAL_SCHEDULE_PATH)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def _save_schedule(df: pd.DataFrame):
    df_sorted = df.sort_values(["date", "game_id"]).reset_index(drop=True)
    df_sorted.to_parquet(CANONICAL_SCHEDULE_PATH, index=False)
    logger.success(f"Canonical schedule updated → {CANONICAL_SCHEDULE_PATH}")


def _save_metadata(df: pd.DataFrame):
    meta = {
        "rows": int(len(df)),
        "min_date": df["date"].min().isoformat() if not df.empty else None,
        "max_date": df["date"].max().isoformat() if not df.empty else None,
        "seasons": (
            sorted(df["season"].dropna().unique().tolist())
            if "season" in df.columns
            else []
        ),
    }
    CANONICAL_METADATA_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info(f"Canonical schedule metadata written → {CANONICAL_METADATA_PATH}")


def ingest_dates(dates: Iterable[date]) -> pd.DataFrame:
    """
    Ingests one or more dates into the canonical schedule snapshot.
    Returns the updated schedule dataframe.
    """
    _ensure_dirs()
    existing = _load_existing_schedule()

    new_rows: list[pd.DataFrame] = []

    for d in dates:
        raw = fetch_scoreboard(d)
        if raw.empty:
            logger.warning(f"No games returned for {d}, skipping.")
            continue

        validated = validate_ingestion_dataframe(raw)
        new_rows.append(validated)

    if not new_rows:
        logger.warning("No new games ingested for the provided dates.")
        return existing

    new_df = pd.concat(new_rows, ignore_index=True)

    if existing.empty:
        merged = new_df
    else:
        merged = pd.concat([existing, new_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=["game_id"]).reset_index(drop=True)

    _save_schedule(merged)
    _save_metadata(merged)

    return merged


def run_today_ingestion(today: date | None = None):
    """
    Ingests games for yesterday and today into the canonical snapshot.
    Intended to be called by the daily orchestrator.
    """
    today = today or date.today()
    yesterday = today - timedelta(days=1)

    logger.info(f"Running today ingestion for [{yesterday}, {today}]")
    schedule_before = _load_existing_schedule()
    rows_before = len(schedule_before)

    updated = ingest_dates([yesterday, today])
    rows_after = len(updated)

    if rows_after == rows_before:
        logger.warning("No games found or ingested for yesterday or today.")
    else:
        logger.info(f"Ingestion added {rows_after - rows_before} new games.")
