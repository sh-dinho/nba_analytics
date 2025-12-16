# ============================================================
# File: src/prediction/train.py
# Purpose: Train NBA ML model and save to disk
# ============================================================

import pickle
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from loguru import logger


def train_model(historical_schedule: pd.DataFrame, model_path: str):
    """
    Train a RandomForestClassifier using historical NBA schedule.

    Args:
        historical_schedule: DataFrame containing features and 'WIN' target
        model_path: Path to save trained model

    Returns:
        Trained RandomForestClassifier
    """
    # Select feature columns
    feature_cols = [
        col for col in historical_schedule.columns if col.startswith("feat_")
    ]
    if not feature_cols:
        raise ValueError("No feature columns found for training.")

    # Target column
    if "WIN" not in historical_schedule.columns:
        raise ValueError("Target column 'WIN' not found in historical data.")

    X = historical_schedule[feature_cols]
    y = historical_schedule["WIN"]

    # Ensure at least 2 classes
    if y.nunique() < 2:
        raise ValueError("Cannot train model: Need at least 2 classes in target.")

    # Split for internal validation (optional)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    logger.info(
        f"Model trained on {len(X_train)} samples, validated on {len(X_test)} samples"
    )

    # Save model
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {model_path}")

    return model
