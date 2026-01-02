from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Verify Canonical Snapshot
# File: src/scripts/verify_nba_data.py
# Author: Sadiq
# ============================================================

import pandas as pd
from datetime import datetime, timedelta
from loguru import logger

from src.config.paths import LONG_SNAPSHOT
from src.ingestion.validator.checks import (
    find_asymmetry,
    find_score_mismatches,
)


REQUIRED_COLUMNS = {
    "game_id",
    "team",
    "opponent",
    "date",
    "points",
    "opponent_points",
}


def verify_snapshot():
    print("\n=== VERIFYING CANONICAL LONG SNAPSHOT ===")

    # --------------------------------------------------------
    # Load snapshot
    # --------------------------------------------------------
    if not LONG_SNAPSHOT.exists():
        logger.error(f"No LONG_SNAPSHOT found at {LONG_SNAPSHOT}")
        return

    logger.info(f"Loading LONG_SNAPSHOT from {LONG_SNAPSHOT}...")
    try:
        df = pd.read_parquet(LONG_SNAPSHOT, engine="pyarrow")
    except Exception as e:
        logger.error(f"Failed to read LONG_SNAPSHOT: {e}")
        return

    print(f"Total Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    # --------------------------------------------------------
    # Schema validation
    # --------------------------------------------------------
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        logger.error(f"❌ Missing required columns: {missing}")
        return
    else:
        logger.success("✅ All required columns present.")

    # --------------------------------------------------------
    # Season info (optional)
    # --------------------------------------------------------
    if "season" in df.columns:
        seasons = df["season"].dropna().unique()
        if len(seasons) > 0:
            print(f"Seasons: {len(seasons)} ({min(seasons)} → {max(seasons)})")
        else:
            print("Seasons: [present but empty]")
    else:
        print("Seasons: [no season column present]")

    # --------------------------------------------------------
    # Integrity Checks
    # --------------------------------------------------------
    print("\n--- Integrity Checks ---")

    # Duplicate (game_id, team)
    dup_count = df.duplicated(subset=["game_id", "team"]).sum()
    print(f"Duplicate (game_id, team): {dup_count} {'[FAIL]' if dup_count else '[OK]'}")

    # Opponent symmetry
    asym_ids = find_asymmetry(df)
    print(f"Opponent Symmetry Errors: {len(asym_ids)} {'[FAIL]' if asym_ids else '[OK]'}")
    if len(asym_ids) > 0:
        print(f"  Sample: {asym_ids[:5].tolist()}")

    # Score symmetry
    score_mismatch_ids = find_score_mismatches(df)
    print(f"Score Mismatches: {len(score_mismatch_ids)} {'[WARN]' if score_mismatch_ids else '[OK]'}")

    # --------------------------------------------------------
    # Freshness Check
    # --------------------------------------------------------
    try:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        last_date = df["date"].max()
        print(f"\nLast Ingested Date: {last_date.date()}")

        if datetime.utcnow().date() - last_date.date() > timedelta(days=2):
            logger.warning("⚠️ Snapshot appears stale (older than 2 days).")
    except Exception:
        logger.error("❌ Failed to parse 'date' column for freshness check.")

    # --------------------------------------------------------
    # Null season check
    # --------------------------------------------------------
    if "season" in df.columns and df["season"].isna().any():
        logger.error("CRITICAL: Null values found in 'season' column!")

    # --------------------------------------------------------
    # Numeric sanity checks
    # --------------------------------------------------------
    print("\n--- Numeric Sanity Checks ---")
    numeric_cols = ["points", "opponent_points"]

    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            logger.error(f"❌ Column {col} is not numeric dtype.")
            continue

        if (df[col] < 0).any():
            logger.error(f"❌ Negative values found in {col}.")

        if df[col].max() > 200:
            logger.warning(f"⚠️ Suspiciously high values in {col}: max={df[col].max()}")

    # --------------------------------------------------------
    # Preview latest rows
    # --------------------------------------------------------
    print("\n--- Latest 5 Rows ---")
    try:
        print(df.sort_values("date").tail(5))
    except Exception:
        print(df.tail(5))

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    print("\n=== SNAPSHOT VERIFICATION COMPLETE ===\n")


if __name__ == "__main__":
    verify_snapshot()
