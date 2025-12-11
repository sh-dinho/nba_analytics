#!/usr/bin/env python
# ============================================================
# File: src/scripts/generate_features.py
# Purpose: Generate features from enriched schedule (prefers WL outcomes)
# ============================================================

import os
import pandas as pd
from src.features.feature_engineering import generate_features_for_games
from src.utils.logging_config import configure_logging

# Prefer enriched file if available
ENRICHED_FILE = "data/cache/historical_schedule_with_results.parquet"
DEFAULT_FILE = "data/cache/historical_schedule.parquet"

FEATURES_FILE = "data/cache/features_full.parquet"
FEATURES_CSV = "data/csv/features_full.csv"


def main():
    logger = configure_logging(name="scripts.generate_features")
    logger.info("Starting feature generation from historical NBA games...")

    # Choose enriched file if it exists
    input_file = ENRICHED_FILE if os.path.exists(ENRICHED_FILE) else DEFAULT_FILE
    logger.info("Loading schedule from %s", input_file)

    df = pd.read_parquet(input_file)
    logger.info("Loaded %d historical games.", len(df))

    logger.info("Generating features for the games...")
    features = generate_features_for_games(df)

    if features is None or features.empty:
        logger.error("No features generated.")
        return

    os.makedirs(os.path.dirname(FEATURES_FILE), exist_ok=True)
    os.makedirs(os.path.dirname(FEATURES_CSV), exist_ok=True)

    features.to_parquet(FEATURES_FILE, index=False)
    features.to_csv(FEATURES_CSV, index=False)

    logger.info("Features saved to %s (shape: %s)", FEATURES_FILE, features.shape)
    logger.info("Features CSV saved to %s", FEATURES_CSV)
    logger.info("Feature generation process completed successfully.")


if __name__ == "__main__":
    main()
