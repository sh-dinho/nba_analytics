from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v5
# Script: Build Merged Dataset (Scores Only)
# Author: Sadiq
#
# This script:
#   • Loads unified game results from games_master.csv
#   • Normalizes team names
#   • Adds season tagging
#   • Dedupes + sorts
#   • Saves merged_games.csv
#
# No betting lines. No external odds ingestion.
# ============================================================

import pandas as pd
from pathlib import Path
from loguru import logger

from src.utils.team_names import normalize_team


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def clean_team_name(name: str) -> str:
    """Remove symbols, seeds, and whitespace."""
    if not isinstance(name, str):
        return name
    return (
        name.replace("*", "")
        .replace("†", "")
        .strip()
    )


# ------------------------------------------------------------
# Main merge function
# ------------------------------------------------------------
def build_merged_dataset():
    raw_dir = Path("data/raw")
    games_path = raw_dir / "games_master.csv"

    if not games_path.exists():
        raise FileNotFoundError("❌ Missing games_master.csv. Run load_games_multi_source first.")

    logger.info("📥 Loading unified game results...")
    games = pd.read_csv(games_path, parse_dates=["date"])

    # Normalize team names
    games["home_team"] = games["home_team"].apply(clean_team_name).apply(normalize_team)
    games["away_team"] = games["away_team"].apply(clean_team_name).apply(normalize_team)

    # Add season column
    games["season"] = games["date"].dt.year.where(
        games["date"].dt.month >= 10,
        games["date"].dt.year - 1,
    )

    # Sort + dedupe
    merged = games.drop_duplicates(subset=["date", "home_team", "away_team"])
    merged = merged.sort_values("date").reset_index(drop=True)

    # Save
    out_path = raw_dir / "merged_games.csv"
    merged.to_csv(out_path, index=False)

    logger.success(f"🎉 Merged dataset saved → {out_path}")
    logger.success(f"📊 Total games: {len(merged)}")

    return out_path


# ------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------
if __name__ == "__main__":
    build_merged_dataset()
