from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v5
# Script: Convert Unified Game Results → Canonical Format
# Author: Sadiq
#
# Features:
#   • Auto-detect CSV inside a folder
#   • Validate required columns (scores only)
#   • Normalize team names
#   • Filter from 2016 season onward
#   • Produce canonical long + schedule snapshots
# ============================================================

import pandas as pd
from pathlib import Path
from loguru import logger

from src.config.paths import (
    CANONICAL_DIR,
    LONG_SNAPSHOT,
    DAILY_SCHEDULE_SNAPSHOT,
)
from src.utils.team_names import normalize_team


REQUIRED_COLUMNS = {
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
}


# ------------------------------------------------------------
# Auto-detect CSV file
# ------------------------------------------------------------
def find_csv(path: str | Path) -> Path:
    path = Path(path)

    if path.is_file() and path.suffix.lower() == ".csv":
        return path

    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    csv_files = list(path.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in folder: {path}")

    if len(csv_files) > 1:
        logger.warning(f"Multiple CSV files found. Using: {csv_files[0].name}")

    return csv_files[0]


# ------------------------------------------------------------
# Validate dataset columns
# ------------------------------------------------------------
def validate_columns(df: pd.DataFrame):
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}\n"
            f"Columns found: {list(df.columns)}"
        )


# ------------------------------------------------------------
# Main converter
# ------------------------------------------------------------
def convert_public_dataset(path: str):
    csv_path = find_csv(path)
    logger.info(f"📥 Loading dataset: {csv_path}")

    df = pd.read_csv(csv_path)

    validate_columns(df)

    # --------------------------------------------------------
    # 1. Convert date + filter from 2016 season onward
    # --------------------------------------------------------
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= pd.Timestamp("2016-10-01")].reset_index(drop=True)

    logger.info(f"📅 Filtered to {len(df)} games from 2016–present")

    # --------------------------------------------------------
    # 2. Normalize team names
    # --------------------------------------------------------
    df["home_team"] = df["home_team"].apply(normalize_team)
    df["away_team"] = df["away_team"].apply(normalize_team)

    before = len(df)
    df = df.dropna(subset=["home_team", "away_team"]).reset_index(drop=True)
    after = len(df)

    if after < before:
        logger.warning(f"⚠️ Dropped {before - after} rows due to unknown team names")

    # --------------------------------------------------------
    # 3. Compute game-level stats
    # --------------------------------------------------------
    df["total_points"] = df["home_score"] + df["away_score"]
    df["margin"] = df["home_score"] - df["away_score"]

    # --------------------------------------------------------
    # 4. Generate stable game_id
    # --------------------------------------------------------
    df["game_id"] = (
        df["date"].dt.strftime("%Y-%m-%d")
        + "_"
        + df["away_team"]
        + "_"
        + df["home_team"]
    )

    # --------------------------------------------------------
    # 5. Build long-format canonical dataset
    # --------------------------------------------------------
    long_rows = []

    for _, row in df.iterrows():
        # Away row
        long_rows.append({
            "game_id": row["game_id"],
            "date": row["date"],
            "team": row["away_team"],
            "opponent": row["home_team"],
            "is_home": 0,
            "score": row["away_score"],
            "opp_score": row["home_score"],
            "win": int(row["away_score"] > row["home_score"]),
            "margin": row["away_score"] - row["home_score"],
            "total_points": row["total_points"],
        })

        # Home row
        long_rows.append({
            "game_id": row["game_id"],
            "date": row["date"],
            "team": row["home_team"],
            "opponent": row["away_team"],
            "is_home": 1,
            "score": row["home_score"],
            "opp_score": row["away_score"],
            "win": int(row["home_score"] > row["away_score"]),
            "margin": row["home_score"] - row["away_score"],
            "total_points": row["total_points"],
        })

    long_df = pd.DataFrame(long_rows)

    # --------------------------------------------------------
    # 6. Save outputs
    # --------------------------------------------------------
    CANONICAL_DIR.mkdir(parents=True, exist_ok=True)

    long_df.to_parquet(LONG_SNAPSHOT, index=False)
    df.to_parquet(DAILY_SCHEDULE_SNAPSHOT, index=False)

    logger.success("🎉 Conversion complete!")
    logger.success(f"📄 Long snapshot → {LONG_SNAPSHOT}")
    logger.success(f"📄 Schedule snapshot → {DAILY_SCHEDULE_SNAPSHOT}")


# ------------------------------------------------------------
# CLI Entrypoint
# ------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert unified NBA dataset to canonical format")
    parser.add_argument("--path", type=str, required=True, help="CSV file or folder containing CSV")

    args = parser.parse_args()
    convert_public_dataset(args.path)
