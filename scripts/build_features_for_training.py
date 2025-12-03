# ============================================================
# File: scripts/build_features_for_training.py
# Purpose: Build training features from historical games
# ============================================================

import pandas as pd
from core.config import HISTORICAL_GAMES_FILE, TRAINING_FEATURES_FILE
from core.log_config import setup_logger

logger = setup_logger("build_features_training")


def build_features_for_training() -> str:
    """
    Build training features from historical games.
    Adds multiple labels:
      - label (binary win/loss)
      - margin (point differential)
      - overtime (flag if game went to OT)
      - outcome_category (categorical: 'home_blowout', 'home_close', 'away_blowout', 'away_close')
    Saves to TRAINING_FEATURES_FILE and returns path.
    """
    logger.info("Loading historical games...")
    df = pd.read_csv(HISTORICAL_GAMES_FILE)

    # Example feature engineering: rolling averages
    features = pd.DataFrame({
        "game_id": df["game_id"],
        "home_team": df["home_team"],
        "away_team": df["away_team"],
        "home_avg_pts": df["home_pts"].rolling(5, min_periods=1).mean(),
        "away_avg_pts": df["away_pts"].rolling(5, min_periods=1).mean(),
        "home_avg_ast": df["home_ast"].rolling(5, min_periods=1).mean(),
        "away_avg_ast": df["away_ast"].rolling(5, min_periods=1).mean(),
        "home_avg_reb": df["home_reb"].rolling(5, min_periods=1).mean(),
        "away_avg_reb": df["away_reb"].rolling(5, min_periods=1).mean(),
        "home_avg_games_played": df.groupby("home_team").cumcount() + 1,
        "away_avg_games_played": df.groupby("away_team").cumcount() + 1,
    })

    # ✅ Labels
    # Binary win/loss
    features["label"] = (df["home_pts"] > df["away_pts"]).astype(int)

    # Margin of victory
    features["margin"] = (df["home_pts"] - df["away_pts"]).astype(int)

    # Overtime flag (assuming 'overtime' column exists in raw data, else derive from minutes)
    if "overtime" in df.columns:
        features["overtime"] = df["overtime"].astype(int)
    else:
        features["overtime"] = 0  # fallback if not available

    # Outcome category
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

    # Save to CSV
    features.to_csv(TRAINING_FEATURES_FILE, index=False)
    logger.info(f"✅ Training features built ({len(features)} rows) → {TRAINING_FEATURES_FILE}")

    return str(TRAINING_FEATURES_FILE)


if __name__ == "__main__":
    build_features_for_training()