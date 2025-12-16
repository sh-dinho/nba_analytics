"""
File: model.py
Path: src/prediction
Purpose: Provides functions to train, save, and use ML models
         for NBA game outcome prediction.
"""

import logging
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

logger = logging.getLogger(__name__)


# ----------------------
# Train Model
# ----------------------
def train_model(x: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
    """
    Trains a RandomForestClassifier on the provided features and target.

    Args:
        x (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector.

    Returns:
        RandomForestClassifier: Trained model.
    """
    if x.empty or y.empty:
        logger.error("Empty features or target. Cannot train model.")
        raise ValueError("Empty features or target.")

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x, y)
    logger.info("Model training complete.")
    return model


# ----------------------
# Save Model
# ----------------------
def save_model(model: RandomForestClassifier, path: Path):
    """
    Saves the trained model to disk using joblib.

    Args:
        model (RandomForestClassifier): Trained model.
        path (Path): File path to save the model.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"Saved model to {path}")


# ----------------------
# Load Model
# ----------------------
def load_model(path: Path) -> RandomForestClassifier:
    """
    Loads a trained model from disk.

    Args:
        path (Path): Path to saved model.

    Returns:
        RandomForestClassifier: Loaded model.
    """
    if not path.exists():
        logger.error(f"Model file not found: {path}")
        raise FileNotFoundError(f"Model file not found: {path}")

    model = joblib.load(path)
    logger.info(f"Loaded model from {path}")
    return model


# ----------------------
# Predict Games
# ----------------------
def predict_games(
    model: RandomForestClassifier, features: pd.DataFrame
) -> pd.DataFrame:
    """
    Predicts the outcome for upcoming NBA games.

    Args:
        model (RandomForestClassifier): Trained model.
        features (pd.DataFrame): Features of upcoming games.

    Returns:
        pd.DataFrame: Input features with an added column 'predicted_home_win'.
    """
    if features.empty:
        logger.warning("No features provided for prediction.")
        return pd.DataFrame()

    predictions = model.predict(features)
    result = features.copy()
    result["predicted_home_win"] = predictions
    logger.info(f"Predicted {len(predictions)} games.")
    return result
