# ============================================================
# File: scripts/build_features.py
# Purpose: Build features for training and prediction
# ============================================================

import argparse
import pandas as pd
from core.config import (
    HISTORICAL_GAMES_FILE,
    TRAINING_FEATURES_FILE,
    NEW_GAMES_FILE,
    NEW_GAMES_FEATURES_FILE,
)
from core.log_config import setup_logger

logger = setup_logger("build_features")


def build_features(rounds: int = 10, training: bool = True) -> str:
    """
    Build features for training or new games.
    Adds labels (label, margin, outcome_category) for training data.
    Generates synthetic game_id if missing in new games.
    """
    if training:
        logger.info("Loading historical games...")
        df = pd.read_csv(HISTORICAL_GAMES_FILE)

        # Example rolling averages
        features = pd.DataFrame({
            "game_id": df.get("game_id", pd.Series(range(len(df)))),
            "home_team": df["home_team"],
            "away_team": df["away_team"],
            "home_avg_pts": df["home_pts"].rolling(rounds, min_periods=1).mean(),
            "away_avg_pts": df["away_pts"].rolling(rounds, min_periods=1).mean(),
            "home_avg_ast": df["home_ast"].rolling(rounds, min_periods=1).mean(),
            "away_avg_ast": df["away_ast"].rolling(rounds, min_periods=1).mean(),
            "home_avg_reb": df["home_reb"].rolling(rounds, min_periods=1).mean(),
            "away_avg_reb": df["away_reb"].rolling(rounds, min_periods=1).mean(),
            "home_games_played": df.groupby("home_team").cumcount() + 1,
            "away_games_played": df.groupby("away_team").cumcount() + 1,
        })

        # ✅ Labels
        features["label"] = (df["home_pts"] > df["away_pts"]).astype(int)
        features["margin"] = (df["home_pts"] - df["away_pts"]).astype(int)

        def categorize(row):
            if row["margin"] >= 10 and row["label"] == 1:
                return "home_blowout"
            elif row["margin"] < 10 and row["label"] == 1:
                return "home_close"
            elif row["margin"] <= -10 and row["label"] == 0:
                return "away_blowout"
            else:
                return "away_close"

        features["outcome_category"] = features.apply(categorize, axis=1)

        # Save
        features.to_csv(TRAINING_FEATURES_FILE, index=False)
        logger.info(f"✅ Training features built ({len(features)} rows) → {TRAINING_FEATURES_FILE}")
        return str(TRAINING_FEATURES_FILE)

    else:
        logger.info("Loading new games...")
        df = pd.read_csv(NEW_GAMES_FILE)

        # Generate synthetic game_id if missing
        if "game_id" in df.columns:
            game_ids = df["game_id"]
        else:
            game_ids = (
                df.get("date", pd.Series(range(len(df)))).astype(str)
                + "_" + df["home_team"] + "_" + df["away_team"]
            )

        features = pd.DataFrame({
            "game_id": game_ids,
            "home_team": df["home_team"],
            "away_team": df["away_team"],
            "home_avg_pts": df["home_pts"].rolling(rounds, min_periods=1).mean(),
            "away_avg_pts": df["away_pts"].rolling(rounds, min_periods=1).mean(),
            "home_avg_ast": df["home_ast"].rolling(rounds, min_periods=1).mean(),
            "away_avg_ast": df["away_ast"].rolling(rounds, min_periods=1).mean(),
            "home_avg_reb": df["home_reb"].rolling(rounds, min_periods=1).mean(),
            "away_avg_reb": df["away_reb"].rolling(rounds, min_periods=1).mean(),
            "home_games_played": df.groupby("home_team").cumcount() + 1,
            "away_games_played": df.groupby("away_team").cumcount() + 1,
        })

        # Save
        features.to_csv(NEW_GAMES_FEATURES_FILE, index=False)
        logger.info(f"✅ New game features built ({len(features)} rows) → {NEW_GAMES_FEATURES_FILE}")
        return str(NEW_GAMES_FEATURES_FILE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build features for training or new games")
    parser.add_argument("--rounds", type=int, default=10, help="Rolling window size")
    parser.add_argument("--training", action="store_true", help="Build training features")
    args = parser.parse_args()

    build_features(rounds=args.rounds, training=args.training)