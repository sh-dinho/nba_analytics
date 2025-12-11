# ============================================================
# File: src/scripts/generate_features.py
# Purpose: Generate training features from historical NBA games
# Project: nba_analysis
# Version: 1.2 (adds dependencies section, ensures CSV dir exists)
#
# Dependencies:
# - logging (standard library)
# - os (standard library)
# - datetime (standard library)
# - pandas
# - src.features.feature_engineering.generate_features_for_games
# ============================================================

import logging
import os
from datetime import datetime
import pandas as pd

from src.features.feature_engineering import generate_features_for_games

# -----------------------------
# CONFIG
# -----------------------------
HISTORICAL_SCHEDULE_FILE = "data/cache/historical_schedule.parquet"
FEATURES_CACHE_FILE = "data/cache/features_full.parquet"
FEATURES_CSV_FILE = "data/csv/features_full.csv"
OUT_DIR = "data/cache"
CSV_DIR = "data/csv"

# -----------------------------
# LOGGING
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# -----------------------------
# MAIN
# -----------------------------
def main():
    logging.info("Starting feature generation from historical NBA games...")

    # Check if the historical schedule file exists
    if not os.path.exists(HISTORICAL_SCHEDULE_FILE):
        logging.error(f"Historical schedule file not found: {HISTORICAL_SCHEDULE_FILE}")
        return

    try:
        # Load the historical games data
        df_games = pd.read_parquet(HISTORICAL_SCHEDULE_FILE)
        logging.info(f"Loaded {len(df_games)} historical games.")
    except Exception as e:
        logging.error(f"Error loading historical games data: {e}")
        return

    # Generate features for the games
    logging.info("Generating features for the games...")
    try:
        features_df = generate_features_for_games(df_games.to_dict(orient="records"))
    except Exception as e:
        logging.error(f"Error generating features: {e}")
        return

    # Check if features were generated
    if features_df.empty:
        logging.warning("No features were generated.")
        return

    # Ensure the output directories exist
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)

    # Save features as Parquet
    try:
        features_df.to_parquet(FEATURES_CACHE_FILE, index=False)
        logging.info(f"Features saved to {FEATURES_CACHE_FILE} (shape: {features_df.shape})")
    except Exception as e:
        logging.error(f"Error saving features as Parquet: {e}")
        return

    # Save features as CSV
    try:
        features_df.to_csv(FEATURES_CSV_FILE, index=False)
        logging.info(f"Features CSV saved to {FEATURES_CSV_FILE}")
    except Exception as e:
        logging.error(f"Error saving features as CSV: {e}")

    logging.info("Feature generation process completed.")

if __name__ == "__main__":
    main()
