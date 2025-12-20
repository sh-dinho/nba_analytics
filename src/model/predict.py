"""
Prediction utilities for NBA Analytics v3.
Handles batch prediction and probability generation.
"""

import pandas as pd
from pathlib import Path
from loguru import logger


def predict_games(
    model, metadata: dict, df_features: pd.DataFrame, output_dir="data/predictions"
):
    """
    Generate predictions for a set of games using a trained model.

    Args:
        model: Trained ML model (e.g., RandomForestClassifier)
        metadata: Model metadata including 'features_used'
        df_features: DataFrame containing feature columns
        output_dir: Directory to save prediction parquet file

    Returns:
        Tuple[pd.DataFrame, Path]: DataFrame with predicted probabilities, path to parquet
    """
    logger.info("Running batch predictions...")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = metadata["features_used"]
    df = df_features.copy()

    # Ensure all features exist
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required feature columns: {missing_cols}")

    # Get model prediction probabilities
    classes = model.classes_
    proba = model.predict_proba(df[feature_cols])

    # Handle single-class models gracefully
    if proba.shape[1] == 1:
        only_class = classes[0]
        df["probability"] = float(only_class)
        logger.warning(
            f"Model trained on a single class ({only_class}). "
            f"All probabilities set to {float(only_class)}."
        )
    else:
        # Normal case: probability of class 1 (home win)
        df["probability"] = proba[:, 1]

    # Save predictions
    timestamp = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"predictions_{timestamp}.parquet"
    df.to_parquet(path, index=False)

    logger.info(f"Saved predictions → {path}")
    return df, path
