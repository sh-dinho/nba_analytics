# ============================================================
# File: scripts/build_features.py
# Purpose: Build training or prediction features
# ============================================================

import argparse
import os
import pandas as pd
from core.config import HISTORICAL_GAMES_FILE, NEW_GAMES_FILE, BASE_DATA_DIR
from core.log_config import setup_logger

logger = setup_logger("build_features")

# Column mapping to normalize different CSV schemas
COLUMN_MAP = {
    "Home": "home_team",
    "Away": "away_team",
    "HomeScore": "home_pts",
    "AwayScore": "away_pts",
    "Date": "date",
    "GameID": "game_id"
}

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns if they differ from expected schema."""
    df = df.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns})
    return df

def build_features(rounds=10, training=False):
    if training:
        logger.info("Loading historical games...")
        df = pd.read_csv(HISTORICAL_GAMES_FILE)
        df = normalize_columns(df)

        # Ensure required columns exist
        required = ["home_team", "away_team", "home_pts", "away_pts"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        features = pd.DataFrame({
            "game_id": df.index,
            "home_team": df["home_team"],
            "away_team": df["away_team"],
            "home_avg_pts": df["home_pts"].rolling(rounds, min_periods=1).mean(),
            "away_avg_pts": df["away_pts"].rolling(rounds, min_periods=1).mean(),
        })

        # Labels
        features["label"] = (df["home_pts"] > df["away_pts"]).astype(int)
        features["margin"] = (df["home_pts"] - df["away_pts"]).astype(int)
        features["outcome_category"] = features["margin"].apply(
            lambda m: "home_blowout" if m >= 10 else
                      "home_close" if m > 0 else
                      "away_close" if m > -10 else
                      "away_blowout"
        )

        out_file = os.path.join(BASE_DATA_DIR, "training_features.csv")
        features.to_csv(out_file, index=False)
        logger.info(f"✅ Training features saved to {out_file} ({len(features)} rows)")

    else:
        logger.info("Loading new games...")
        df = pd.read_csv(NEW_GAMES_FILE)
        df = normalize_columns(df)

        required = ["home_team", "away_team"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        features = pd.DataFrame({
            "game_id": df.index,
            "home_team": df["home_team"],
            "away_team": df["away_team"],
        })

        out_file = os.path.join(BASE_DATA_DIR, "new_games_features.csv")
        features.to_csv(out_file, index=False)
        logger.info(f"✅ New game features saved to {out_file} ({len(features)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build features for training or prediction")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--training", action="store_true")
    args = parser.parse_args()

    build_features(rounds=args.rounds, training=args.training)