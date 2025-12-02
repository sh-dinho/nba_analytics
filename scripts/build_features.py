# File: scripts/build_features.py

import os
import logging
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

REQUIRED_COLS = [
    "PLAYER_NAME",
    "TEAM_ABBREVIATION",
    "AGE",
    "POSITION",
    "GAMES_PLAYED",
    "PTS",
    "AST",
    "REB"
]

def main(n_rounds: int = 1):
    stats_file = "data/player_stats.csv"
    if not os.path.exists(stats_file):
        raise FileNotFoundError(
            f"{stats_file} not found. Run fetch_player_stats.py first or use synthetic fallback."
        )

    logger.info(f"Loading player stats from {stats_file}...")
    df = pd.read_csv(stats_file)

    # Ensure required columns exist
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available: {df.columns.tolist()}")

    logger.info("Building features...")

    # Example engineered features
    df["PTS_per_AST"] = df["PTS"] / df["AST"].replace(0, pd.NA)
    df["REB_rate"] = df["REB"] / df["GAMES_PLAYED"].replace(0, pd.NA)

    # Save features
    os.makedirs("data", exist_ok=True)
    out_file = "data/training_features.csv"
    df.to_csv(out_file, index=False)
    logger.info(f"✅ Features saved to {out_file} ({len(df)} rows)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=1, help="Number of feature-building rounds")
    args = parser.parse_args()
    main(n_rounds=args.rounds)