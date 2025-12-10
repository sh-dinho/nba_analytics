# ============================================================
# File: src/scripts/generate_features.py
# Purpose: Generate training features from historical NBA games
# Project: nba_analysis
# Version: 1.0
# ============================================================

import pandas as pd
import os
import logging
from datetime import datetime
from src.features.feature_engineering import generate_features_for_games

# -----------------------------
# CONFIG
# -----------------------------
HISTORICAL_SCHEDULE_FILE = "data/cache/historical_schedule.parquet"
FEATURES_CACHE_FILE = "data/cache/features_full.parquet"
FEATURES_CSV_FILE = "data/csv/features_full.csv"
OUT_DIR = "data/cache"

# -----------------------------
# LOGGING
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# -----------------------------
# MAIN
# -----------------------------
def main():
    logging.info("Generating features from historical games...")

    if not os.path.exists(HISTORICAL_SCHEDULE_FILE):
        logging.error(f"Historical schedule file not found: {HISTORICAL_SCHEDULE_FILE}")
        return

    df_games = pd.read_parquet(HISTORICAL_SCHEDULE_FILE)
    logging.info(f"Loaded {len(df_games)} historical games")

    # Generate features using your feature_engineering module
    features_df = generate_features_for_games(df_games.to_dict(orient="records"))

    if features_df.empty:
        logging.warning("No features generated")
        return

    os.makedirs(OUT_DIR, exist_ok=True)

    # Save features as Parquet
    features_df.to_parquet(FEATURES_CACHE_FILE, index=False)
    logging.info(f"Features saved to {FEATURES_CACHE_FILE} (shape: {features_df.shape})")

    # Save features as CSV (optional)
    features_df.to_csv(FEATURES_CSV_FILE, index=False)
    logging.info(f"Features CSV saved to {FEATURES_CSV_FILE}")


if __name__ == "__main__":
    main()
