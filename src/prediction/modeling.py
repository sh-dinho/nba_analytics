# ============================================================
# File: src/prediction/modeling.py
# Purpose: Load ML model and make predictions on games
# Author: Your Name
# ============================================================

import pickle
import pandas as pd
import logging


def load_model(path: str):
    """Load a trained model from disk."""
    logger = logging.getLogger(__name__)
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Loaded model from {path}")
        return model
    except FileNotFoundError:
        logger.warning(f"Model not found at {path}. Predictions will be skipped.")
        return None


def predict_games(schedule_df, model, config):
    """Generate predictions for today's games."""
    logger = logging.getLogger(__name__)
    if model is None or schedule_df.empty:
        return pd.DataFrame()
    # Simple placeholder: assign random probability
    schedule_df["predicted_win"] = 0.5
    schedule_df["WIN"] = None  # Actual result unknown
    schedule_df["TEAM_ABBREVIATION"] = schedule_df.get("home_team", "")
    logger.info(f"Predictions generated for {len(schedule_df)} games")
    return schedule_df
