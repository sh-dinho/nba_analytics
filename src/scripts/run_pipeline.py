# ============================================================
# File: src/pipeline/run_pipeline.py
# Purpose: End-to-end pipeline to fetch today's NBA games,
#          generate predictions using a model wrapper,
#          and save results to disk.
# Project: nba_analysis
# Version: 1.1 (adds dependencies section, clarifies logging and model placeholder)
#
# Dependencies:
# - logging (standard library)
# - datetime (standard library)
# - src.utils.io
# - src.utils.logging_config
# - src.utils.validation
# - src.api.nba_api_wrapper.fetch_today_games
# - src.prediction_engine.predictor.Predictor
# ============================================================

import logging
from datetime import datetime

from src.utils import io, logging_config, validation
from src.api.nba_api_wrapper import fetch_today_games
from src.prediction_engine.predictor import Predictor

# -----------------------------
# DATA FETCHING & PROCESSING
# -----------------------------
def fetch_and_process_data():
    logging.info("Fetching today's games...")
    today_games = fetch_today_games()
    if today_games.empty:
        logging.warning("No games today, skipping pipeline.")
        return None
    # Apply necessary transformations
    today_games = today_games.dropna()  # Example cleaning step
    return today_games

# -----------------------------
# PREDICTION GENERATION
# -----------------------------
def generate_predictions(df):
    # TODO: Replace `some_model` with your actual trained model instance
    # For example: some_model = joblib.load("models/nba_xgb.pkl")
    some_model = None  # Placeholder
    if some_model is None:
        raise RuntimeError("No model provided. Please load a trained model before prediction.")
    model = Predictor(some_model)
    predictions = model.predict(df)
    return predictions

# -----------------------------
# RESULTS SAVING
# -----------------------------
def save_results(predictions, path="data/results/predictions.parquet"):
    io.save_dataframe(predictions, path)
    logging.info(f"Predictions saved to {path}")

# -----------------------------
# PIPELINE RUNNER
# -----------------------------
def run_pipeline():
    try:
        # Step 1: Fetch data
        data = fetch_and_process_data()
        if data is None:
            return

        # Step 2: Generate predictions
        predictions = generate_predictions(data)

        # Step 3: Save results
        save_results(predictions)

    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}")

if __name__ == "__main__":
    run_pipeline()
