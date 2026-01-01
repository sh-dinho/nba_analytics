from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Auto Backfill
# File: src/ingestion/maintenance/auto_backfill.py
# Author: Sadiq
#
# Description:
#     Automatically detects and backfills missing dates in the
#     canonical long-format snapshot. Integrates with the
#     orchestrator and validator to ensure safe recovery.
# ============================================================

from datetime import date
import pandas as pd
from loguru import logger

from src.config.paths import LONG_SNAPSHOT
from src.ingestion.maintenance.missing_dates import detect_missing_dates
from src.ingestion.orchestrator import ingest_dates
from src.ingestion.validator.team_game_validator import validate_team_game_df


def auto_backfill(start: date, end: date) -> pd.DataFrame:
    """
    Detect and backfill missing dates between [start, end].

    Returns:
        DataFrame of newly ingested rows.
    """
    logger.info(f"[Backfill] Checking for missing dates between {start} and {end}")

    missing = detect_missing_dates(start, end)
    if not missing:
        logger.success("[Backfill] No missing dates detected.")
        return pd.DataFrame()

    logger.warning(f"[Backfill] Missing {len(missing)} dates: {missing}")

    # Ingest missing dates
    new_rows = ingest_dates(missing)
    if new_rows.empty:
        logger.error("[Backfill] No rows ingested during backfill.")
        return pd.DataFrame()

    # Validate the newly ingested rows
    validate_team_game_df(new_rows, raise_on_error=True)

    # Append to snapshot
    if LONG_SNAPSHOT.exists():
        existing = pd.read_parquet(LONG_SNAPSHOT)
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        combined = new_rows

    combined = combined.drop_duplicates(subset=["game_id", "team"]).sort_values(
        ["date", "game_id", "team"]
    )

    combined.to_parquet(LONG_SNAPSHOT, index=False)
    logger.success(
        f"[Backfill] Backfill complete. Snapshot now has {len(combined)} rows."
    )

    return new_rows