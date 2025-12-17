# ============================================================
# File: src/prediction/modeling.py
# Purpose: Load ML model and make predictions on NBA games
# Author: Your Name
# Date: 2023
# Dependencies:
#     - pandas
#     - pickle
#     - logging
# ============================================================

import pickle
import pandas as pd
import logging

# Configure logging
logger = logging.getLogger(__name__)


# ------------------------------
# Load Model
# ------------------------------
def load_model(path: str):
    """
    Load a trained model from disk.

    Args:
        path (str): Path to the model file.

    Returns:
        model: Loaded machine learning model.
    """
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Model successfully loaded from {path}")
        return model
    except FileNotFoundError:
        logger.warning(f"Model not found at {path}. Predictions will be skipped.")
        return None
    except Exception as e:
        logger.error(f"Failed to load model from {path}. Error: {e}")
        return None


# ------------------------------
# Predict Games
# ------------------------------
def predict_games(schedule_df: pd.DataFrame, model, config) -> pd.DataFrame:
    """
    Generate predictions for today's games.

    Args:
        schedule_df (pd.DataFrame): DataFrame containing game schedule with features.
        model: Trained machine learning model (e.g., RandomForest, XGBoost).
        config: Configuration containing settings for model and features.

    Returns:
        pd.DataFrame: Updated DataFrame with prediction probabilities for home and away teams.
    """
    if model is None or schedule_df.empty:
        logger.warning("No model loaded or schedule is empty. Skipping predictions.")
        return pd.DataFrame()

    # Retrieve features from configuration
    required_features = config.get("required_features", [])

    if not required_features:
        logger.error("No required features provided in configuration. Cannot proceed.")
        return pd.DataFrame()

    # Check if the necessary features are in the schedule DataFrame
    missing_features = [
        feature for feature in required_features if feature not in schedule_df.columns
    ]

    if missing_features:
        logger.warning(f"Missing required features for prediction: {missing_features}")
        # Optionally, fill missing features with default values (e.g., 0)
        schedule_df = schedule_df.copy()
        schedule_df["predicted_win"] = None  # Add prediction column with NaN
        return schedule_df  # Return without predictions for missing features

    # Prepare feature matrix (fill missing values with 0 for simplicity)
    X = schedule_df[required_features].fillna(0)

    # Predict the win probability for the home team
    try:
        schedule_df["predicted_win"] = model.predict_proba(X)[
            :, 1
        ]  # Probability of home team win
        schedule_df["predicted_loss"] = (
            1 - schedule_df["predicted_win"]
        )  # Probability of away team win
        logger.info(f"Predictions generated for {len(schedule_df)} games.")
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return pd.DataFrame()  # Return empty DataFrame in case of failure

    # Add or modify other relevant columns for better understanding
    schedule_df["TEAM_ABBREVIATION"] = schedule_df.get(
        "home_team", ""
    )  # Default to home_team abbreviation if no column
    schedule_df["WIN"] = None  # Actual result (WIN) is unknown, leave as None

    return schedule_df
