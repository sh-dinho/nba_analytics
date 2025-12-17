#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train NBA Random Forest Model

Author: Mohamadou
Date: 2023

Purpose:
    This script trains a Random Forest model on historical NBA game data,
    evaluates the model's performance, and saves the trained model to disk.
    The model predicts game outcomes (win/loss) based on various features extracted from
    historical data, including team performance statistics and game-related features.

Dependencies:
    - pandas
    - scikit-learn
    - loguru

Usage:
    Run the script to train the model with a historical schedule dataset
    and save the trained model to disk with a timestamped filename.
"""

import pickle
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from loguru import logger
from datetime import datetime


def train_model(
    historical_schedule: pd.DataFrame,
    model_path: str,
    n_estimators=100,
    max_depth=10,
    random_state=42,
):
    """
    Train a RandomForestClassifier on historical NBA schedule data.

    Args:
        historical_schedule (pd.DataFrame): DataFrame containing features and target 'WIN'.
        model_path (str): The file path to save the trained model.
        n_estimators (int): Number of trees in the forest (default is 100).
        max_depth (int): Maximum depth of the trees (default is 10).
        random_state (int): Seed for random number generation (default is 42).

    Returns:
        RandomForestClassifier: The trained Random Forest model.

    Raises:
        ValueError: If no feature columns or target 'WIN' column are found in the data.
        ValueError: If the target column does not contain at least 2 classes.
    """
    # Select feature columns (those starting with "feat_")
    feature_cols = [
        col for col in historical_schedule.columns if col.startswith("feat_")
    ]
    if not feature_cols:
        raise ValueError("No feature columns found for training.")

    # Target column
    if "WIN" not in historical_schedule.columns:
        raise ValueError("Target column 'WIN' not found in historical data.")

    # Prepare feature matrix (X) and target vector (y)
    x = historical_schedule[feature_cols]
    y = historical_schedule["WIN"]

    # Ensure there are at least 2 classes in the target column
    if y.nunique() < 2:
        raise ValueError(
            "Target column 'WIN' must contain at least 2 classes for classification."
        )

    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # Initialize and train the RandomForestClassifier
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",  # To handle imbalanced classes
    )

    try:
        model.fit(x_train, y_train)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

    # Log model training info
    logger.info(
        f"Model trained on {len(x_train)} samples, validated on {len(x_test)} samples"
    )

    # Evaluate model performance on test set
    y_pred = model.predict(x_test)
    logger.info("\n" + classification_report(y_test, y_pred))

    # Log feature importances
    importance = pd.Series(model.feature_importances_, index=feature_cols)
    logger.info(
        "Top features based on importance:\n"
        + str(importance.sort_values(ascending=False).head(10))
    )

    # Save the trained model to disk
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {model_path}")

    return model


if __name__ == "__main__":
    # Load historical schedule data
    historical_schedule = pd.read_parquet("data/history/historical_schedule.parquet")

    # Generate a timestamp for model filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_file = f"models/nba_randomforest_{timestamp}.pkl"

    # Train the model and save to disk
    trained_model = train_model(historical_schedule, model_file)
