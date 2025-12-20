"""
Training pipeline for NBA Analytics v3 (function-based Feature Store).
This file includes model training logic and utilities for saving and loading models.
"""

from __future__ import annotations
from datetime import datetime
from typing import List
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from loguru import logger
from src.model.registry import (
    ModelRegistry,
)  # Assuming ModelRegistry class is implemented here


def _select_training_columns(df: pd.DataFrame, target_col: str) -> List[str]:
    """
    Select model input features by excluding identifiers and target.
    """
    exclude = {"game_id", "date", "home_team", "away_team", target_col}
    return [c for c in df.columns if c not in exclude]


def train_model(
    feature_df: pd.DataFrame, registry: ModelRegistry, target_col: str = "homeWin"
) -> dict:
    """
    Train a RandomForestClassifier model on the features provided.

    Args:
        feature_df (pd.DataFrame): The DataFrame containing features and target column for training.
        registry (ModelRegistry): An instance of ModelRegistry to handle saving and versioning the model.
        target_col (str): The column to predict (e.g., homeWin).

    Returns:
        dict: Metadata containing model version and other information.
    """
    logger.info("Starting model training")

    try:
        df = feature_df.copy()
        logger.info(f"Training DataFrame shape: {df.shape}")

        # Ensure required score columns exist for target calculation
        if "home_score" not in df.columns or "away_score" not in df.columns:
            raise ValueError(
                "Feature snapshot must include home_score and away_score "
                "to compute the training target."
            )

        # Build target variable (e.g., homeWin as 1 if home_score > away_score, else 0)
        df[target_col] = (df["home_score"] > df["away_score"]).astype(int)

        # Select feature columns (excluding game identifiers and target)
        feature_cols = _select_training_columns(df, target_col)
        logger.info(f"Training with {len(feature_cols)} features")

        # Train the model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(df[feature_cols], df[target_col])

        # Model versioning and metadata
        version = registry.next_version()  # Get next available model version
        metadata = {
            "version": version,
            "timestamp": datetime.utcnow().isoformat(),
            "features_used": feature_cols,
            "target": target_col,
            "model_type": "RandomForestClassifier",
        }

        # Save the trained model to the registry
        registry.save(model, metadata, version)
        logger.info(f"Model saved to registry (version={version})")

        return metadata

    except Exception as e:
        logger.exception(f"Training failed: {e}")
        raise


# ---------------------------------------------------------
# Helper functions for loading models
# ---------------------------------------------------------
def load_model():
    """
    Load the latest model from the registry.

    Returns:
        model: The latest trained model.
    """
    registry = ModelRegistry()
    model, metadata, version = registry.load_latest()
    if model is None:
        raise FileNotFoundError("No trained model found in registry.")
    return model


def load_metadata():
    """
    Load the metadata of the latest model from the registry.

    Returns:
        dict: Metadata of the latest model.
    """
    registry = ModelRegistry()
    model, metadata, version = registry.load_latest()
    if metadata is None:
        raise FileNotFoundError("No metadata found for the latest model.")
    return metadata
