# ============================================================
# File: src/model/predict.py
# Purpose: Utility for loading trained models and predicting NBA outcomes
# Version: 1.0
# Author: Mohamadou
# Date: December 2025
# ============================================================

import logging
import joblib
import pandas as pd
from pathlib import Path

from src.model.train_model import predict_outcomes

# Configure logger for the predict module
logger = logging.getLogger("model.predict")
logging.basicConfig(level=logging.INFO)


def load_model(model_path: str):
    """
    Load a trained model from disk using joblib.

    Args:
        model_path (str): Path to the saved model file.

    Returns:
        model: The loaded machine learning model, or None if loading failed.
    """
    try:
        model = joblib.load(model_path)
        logger.info(f"Model successfully loaded from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        return None


def predict_schedule(model, schedule_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Predict outcomes for NBA games using a trained model and a schedule DataFrame.

    Args:
        model: The trained machine learning model to use for predictions.
        schedule_df (pd.DataFrame): DataFrame containing game features for prediction.
        config (dict): Configuration dictionary with key "features".

    Returns:
        pd.DataFrame: Updated DataFrame with predicted outcomes and probabilities.
    """
    if model is None or schedule_df.empty:
        logger.warning("Model or schedule data is missing. Skipping predictions.")
        return pd.DataFrame()

    # Ensure required feature columns exist
    if not all(col in schedule_df.columns for col in config["features"]):
        logger.warning(
            f"Required columns {config['features']} are missing in the schedule DataFrame."
        )
        return pd.DataFrame()

    try:
        # Reuse predict_outcomes from train_model.py
        schedule_df = predict_outcomes(model, schedule_df, config)
        logger.info(f"Predicted outcomes for {len(schedule_df)} games.")
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return pd.DataFrame()

    return schedule_df


# Example usage
if __name__ == "__main__":
    # Example path to your saved model
    model_path = "models/rf_model_latest.joblib"

    # Example schedule DataFrame with placeholder features
    schedule_data = {
        "HOME_TEAM_STATS": [0.85, 0.76, 0.90],
        "AWAY_TEAM_STATS": [0.72, 0.65, 0.80],
    }
    schedule_df = pd.DataFrame(schedule_data)

    # Example config
    config = {
        "features": ["HOME_TEAM_STATS", "AWAY_TEAM_STATS"],
        "target": "homeWin",
    }

    # Load the model
    model = load_model(model_path)

    # Predict the outcomes for the given schedule
    if model:
        predictions = predict_schedule(model, schedule_df, config)
        print(predictions)
