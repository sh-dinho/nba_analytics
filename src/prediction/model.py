# ============================================================
# File: model.py
# Path: src/prediction
# Purpose: Provides functions to train, save, and use ML models
#          for NBA game outcome prediction.
# Author: Mohamadou
# Date: 2023
# Dependencies:
#     - pandas
#     - scikit-learn
#     - joblib
#     - logging
# ============================================================

import logging
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from datetime import datetime

# Configure logger
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler("logs/model_training.log"),  # Log to a file
    ],
)
logger = logging.getLogger(__name__)

# Model Parameters (Consider externalizing into a config)
RF_PARAMS = {
    "n_estimators": 100,  # Number of trees in the forest
    "random_state": 42,  # Seed for reproducibility
    "n_jobs": -1,  # Use all available cores for training
    "class_weight": "balanced",  # Handle class imbalance
}


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

    # Train-Test Split for Validation
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    logger.info(
        f"Training model with {x_train.shape[0]} samples and validating with {x_val.shape[0]} samples."
    )

    # Initialize the model
    model = RandomForestClassifier(**RF_PARAMS)

    # Train the model
    model.fit(x_train, y_train)
    logger.info("Model training complete.")

    # Cross-validation
    cv_scores = cross_val_score(model, x, y, cv=5)
    logger.info(
        f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})"
    )

    # Validate the model on the validation set
    accuracy = model.score(x_val, y_val)
    logger.info(f"Model validation accuracy: {accuracy:.4f}")

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

    # Check feature column match
    if set(features.columns) != set(model.feature_importances_):
        logger.error(
            f"Feature mismatch: Expected {model.feature_importances_}, but got {features.columns}"
        )
        raise ValueError("Feature mismatch")

    # Predict outcomes
    predictions = model.predict(features)
    result = features.copy()
    result["predicted_home_win"] = predictions

    # Predict probabilities
    probabilities = model.predict_proba(features)
    result["predicted_home_win_prob"] = probabilities[:, 1]  # Home win probability

    logger.info(f"Predicted {len(predictions)} games.")
    return result
