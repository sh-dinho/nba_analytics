import os
import logging
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

TRAINING_FEATURES_FILE = "data/training_features.csv"

def build_features():
    """
    Build training features from player_stats.csv and game_results.csv.
    Adds engineered features like PTS_per_AST and REB_rate.
    """
    os.makedirs("data", exist_ok=True)

    # Load player stats
    stats_file = "data/player_stats.csv"
    if not os.path.exists(stats_file):
        raise FileNotFoundError(f"{stats_file} not found")

    df = pd.read_csv(stats_file)

    # Add engineered features
    df["PTS_per_AST"] = df["PTS"] / df["AST"].replace(0, 1)
    df["REB_rate"] = df["REB"] / df["GAMES_PLAYED"].replace(0, 1)

    # Add synthetic labels if game_results.csv missing
    results_file = "data/game_results.csv"
    if os.path.exists(results_file):
        results = pd.read_csv(results_file)
        df = df.merge(results, on="PLAYER_NAME", how="left")
    else:
        logger.warning("⚠️ No game_results.csv found. Adding synthetic labels.")
        df["home_win"] = [1 if i % 2 == 0 else 0 for i in range(len(df))]

    df.to_csv(TRAINING_FEATURES_FILE, index=False)
    logger.info(f"✅ Features saved to {TRAINING_FEATURES_FILE} ({len(df)} rows)")

def build_features_for_new_games(new_games_file: str):
    """
    Build features for prediction from new_games.csv.
    Ensures engineered features match training.
    """
    if not os.path.exists(new_games_file):
        raise FileNotFoundError(f"{new_games_file} not found")

    df = pd.read_csv(new_games_file)

    # Add engineered features
    df["PTS_per_AST"] = df["PTS"] / df["AST"].replace(0, 1)
    df["REB_rate"] = df["REB"] / df["GAMES_PLAYED"].replace(0, 1)

    return df

if __name__ == "__main__":
    build_features()