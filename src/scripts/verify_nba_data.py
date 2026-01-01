from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Verify Canonical Snapshot
# File: src/scripts/verify_nba_data.py
# Author: Sadiq
#
# Description:
#     Validates the canonical LONG_SNAPSHOT:
#       • row/column integrity
#       • duplicate (game_id, team)
#       • opponent symmetry
#       • score symmetry
#       • season completeness (if available)
#       • freshness
#       • preview latest rows
# ============================================================

import pandas as pd
from loguru import logger

from src.config.paths import LONG_SNAPSHOT
from src.ingestion.validator.checks import (
    find_asymmetry,
    find_score_mismatches,
)


def verify_snapshot():
    if not LONG_SNAPSHOT.exists():
        logger.error(f"No LONG_SNAPSHOT found at {LONG_SNAPSHOT}")
        return

    logger.info(f"Loading LONG_SNAPSHOT from {LONG_SNAPSHOT}...")
    df = pd.read_parquet(LONG_SNAPSHOT, engine="pyarrow")

    print("\n=== CANONICAL SNAPSHOT REPORT ===")
    print(f"Total Rows: {len(df):,}")

    # Season info (optional)
    if "season" in df.columns:
        seasons = df["season"].dropna().unique()
        print(f"Seasons: {len(seasons)} ({min(seasons)} → {max(seasons)})")
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

    # Freshness
    last_date = pd.to_datetime(df["date"]).max()
    print(f"\nLast Ingested Date: {last_date.date()}")

    # Null season check
    if "season" in df.columns and df["season"].isna().any():
        print("CRITICAL: Null values found in 'season' column!")

    # Preview
    print("\n--- Latest 5 Rows ---")
    print(df.sort_values("date").tail(5))

    print("\n=== DONE ===")


if __name__ == "__main__":
    verify_snapshot()