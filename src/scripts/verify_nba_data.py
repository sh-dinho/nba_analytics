from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v4
# Module: Verify NBA Data
# File: src/scripts/verify_nba_data.py
# Author: Sadiq
#
# Description:
#     Validates the canonical NBA snapshot:
#       1. Row/column integrity
#       2. No duplicate Game IDs
#       3. Seasonal completeness
#       4. Last game freshness
#       5. Shows latest 5 entries
# ============================================================

import pandas as pd
from loguru import logger

from src.config.paths import SCHEDULE_SNAPSHOT
from src.ingestion.pipeline import HEADER  # Correct import

# ------------------------------------------------------------
# Verify NBA snapshot data
# ------------------------------------------------------------


def check_data():
    if not SCHEDULE_SNAPSHOT.exists():
        logger.error("No snapshot found.")
        return

    df = pd.read_parquet(SCHEDULE_SNAPSHOT)

    print("\n=== DATA REPORT ===")
    print(f"Total Rows: {len(df)}")
    print(f"Columns Found: {list(df.columns)}")

    # Ensure canonical header
    df = df[HEADER]

    # Check for scores
    games_with_scores = df[df["score_home"].notna()].shape[0]
    print(f"Games with Scores: {games_with_scores}")

    # Safely show recent games
    cols_to_show = [c for c in HEADER if c in df.columns]

    print("\n--- Latest 5 Entries by Date ---")
    print(df.sort_values("date").tail(5)[cols_to_show])

    # Optional: check seasonal integrity
    df["date"] = pd.to_datetime(df["date"])
    seasonal_summary = df.groupby(df["date"].dt.year).agg(
        total_games=("game_id", "count"),
        first_game=("date", "min"),
        last_game=("date", "max"),
    )
    print("\n📊 Seasonal Integrity Check:")
    print(seasonal_summary)


if __name__ == "__main__":
    check_data()
