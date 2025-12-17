# ============================================================
# File: src/model/train_model.py
# Purpose: Model training and prediction (RandomForest) v2.0
# Version: 2.0
# Author: Your Team
# Date: December 2025
# ============================================================

from pathlib import Path
from datetime import datetime
import logging
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def train_model(df: pd.DataFrame, config: dict, model_dir="models"):
    """
    Train a RandomForestClassifier on the given DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Training dataset containing features and target column.
    config : dict
        Configuration dictionary with keys:
            - "features": list of feature column names
            - "target": target column name
            - "n_estimators": number of trees
            - "max_depth": maximum depth of trees
            - "random_state": random seed
    model_dir : str
        Directory to save trained model artifacts.

    Returns
    -------
    model : RandomForestClassifier
        Trained model instance.
    model_path : Path
        Path to the saved model artifact.
    """
    X = df[config["features"]]
    y = df[config["target"]]

    model = RandomForestClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        random_state=config.get("random_state", 42),
    )
    model.fit(X, y)

    Path(model_dir).mkdir(parents=True, exist_ok=True)
    model_path = (
        Path(model_dir) / f"rf_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
    )
    joblib.dump(model, model_path)

    logging.info(f"Model trained and saved to {model_path}")
    return model, model_path


def predict_outcomes(model, df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Add predictions and probabilities to the DataFrame.

    Parameters
    ----------
    model : RandomForestClassifier
        Trained model instance.
    df : pd.DataFrame
        Dataset containing features for prediction.
    config : dict
        Configuration dictionary with key "features".

    Returns
    -------
    df : pd.DataFrame
        DataFrame with added columns:
            - "predicted_outcome": binary predictions
            - "predicted_prob": probability of positive class (if available)
    """
    X = df[config["features"]]

    # Predicted class labels
    df["predicted_outcome"] = model.predict(X)

    # Predicted probabilities
    proba = model.predict_proba(X)
    if proba.shape[1] == 1:
        # Only one class was present during training
        # Assign probability 1.0 for the predicted class
        df["predicted_prob"] = proba[:, 0]
    else:
        # Two classes: take probability of positive class (assumed label 1)
        df["predicted_prob"] = proba[:, 1]

    return df
