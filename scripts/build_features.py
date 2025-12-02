import os
import logging
import pandas as pd
from config import PLAYER_STATS_FILE, GAME_RESULTS_FILE, TRAINING_FEATURES_FILE

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

REQUIRED_COLS = ["PLAYER_NAME", "TEAM_ABBREVIATION", "AGE", "POSITION", "GAMES_PLAYED", "PTS", "AST", "REB"]

def main(n_rounds: int = 1):
    if not os.path.exists(PLAYER_STATS_FILE):
        raise FileNotFoundError(f"{PLAYER_STATS_FILE} not found. Run fetch_player_stats.py first.")

    logger.info(f"Loading player stats from {PLAYER_STATS_FILE}...")
    df = pd.read_csv(PLAYER_STATS_FILE)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    logger.info("Building features...")

    # Safe feature engineering
    df["PTS_per_AST"] = df["PTS"] / df["AST"].replace(0, pd.NA)
    df["PTS_per_AST"] = df["PTS_per_AST"].fillna(0)

    df["REB_rate"] = df["REB"] / df["GAMES_PLAYED"].replace(0, pd.NA)
    df["REB_rate"] = df["REB_rate"].fillna(0)

    # Merge outcomes
    if os.path.exists(GAME_RESULTS_FILE):
        logger.info("Merging real game outcomes...")
        results = pd.read_csv(GAME_RESULTS_FILE)
        if "home_win" in results.columns:
            df["home_win"] = results["home_win"]
        else:
            logger.warning("⚠️ game_results.csv missing 'home_win'. Adding synthetic labels.")
            df["home_win"] = (df.index % 2 == 0).astype(int)
    else:
        logger.warning("⚠️ No game_results.csv found. Adding synthetic labels.")
        df["home_win"] = (df.index % 2 == 0).astype(int)

    os.makedirs(os.path.dirname(TRAINING_FEATURES_FILE), exist_ok=True)
    df.to_csv(TRAINING_FEATURES_FILE, index=False)
    logger.info(f"✅ Features saved to {TRAINING_FEATURES_FILE} ({len(df)} rows)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=1)
    args = parser.parse_args()
    main(n_rounds=args.rounds)