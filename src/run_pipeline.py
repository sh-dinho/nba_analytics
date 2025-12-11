#!/usr/bin/env python
# ============================================================
# File: src/run_pipeline.py
# Purpose: Main script to run the data pipeline and prediction process
# Project: nba_analysis
# Version: 1.3 (adds dependencies section + audit logging + clearer error handling)
#
# Dependencies:
# - logging (standard library)
# - os (standard library)
# - src.api.nba_api_wrapper
# - src.features.feature_engineering
# - src.model_training.trainer
# - src.prediction_engine.predictor
# - src.utils.io
# ============================================================

import logging
import os

from src.api.nba_api_wrapper import fetch_season_games, fetch_today_games
from src.features.feature_engineering import generate_features_for_games
from src.model_training.trainer import ModelTrainer
from src.prediction_engine.predictor import NBAPredictor
from src.utils.io import save_dataframe

# -----------------------------
# LOGGING CONFIGURATION
# -----------------------------
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "pipeline.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),              # Console output
        logging.FileHandler(LOG_FILE, mode="a")  # Persistent audit log
    ],
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("run_pipeline")


def run_data_pipeline():
    """Run the full data pipeline for NBA predictions."""
    try:
        # Step 1: Fetch today's games
        logger.info("Fetching today's NBA games...")
        today_games = fetch_today_games()

        if today_games.empty:
            logger.warning("No games available today.")
            return

        # Step 2: Generate features for prediction
        logger.info("Generating features for today's games...")
        features = generate_features_for_games(today_games)

        if features.empty:
            logger.warning("No features generated.")
            return

        # Step 3: Load the trained model for prediction
        logger.info("Loading the trained model...")
        model = NBAPredictor(model_path="models/trained_model.pkl")

        # Step 4: Make predictions
        logger.info("Making predictions...")
        predictions = model.predict_label(features)

        # Step 5: Track and log the predictions
        logger.info("Tracking predictions...")
        predictions_df = today_games.copy()
        predictions_df["prediction"] = predictions

        # Save predictions
        save_dataframe(predictions_df, "results/today_predictions.csv")
        logger.info(
            f"Predictions saved to results/today_predictions.csv "
            f"(rows: {len(predictions_df)})"
        )

    except Exception as e:
        logger.exception("Error in running the pipeline")


if __name__ == "__main__":
    run_data_pipeline()
