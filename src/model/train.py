"""
Model Training Pipeline
-----------------------
Loads long-format canonical data, builds ML features, trains a model,
and saves it to the model registry.

This version:
- Assumes all games from nba_api are completed ("final")
- Adds a synthetic `status` column
- Removes the broken filter `features_df["status"] == "final"`
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from datetime import datetime

import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.config.paths import LONG_SNAPSHOT, MODEL_REGISTRY_DIR
from src.features.builder import FeatureBuilder


# ---------------------------------------------------------
# Load long-format data
# ---------------------------------------------------------


def _load_long_data() -> pd.DataFrame:
    if not LONG_SNAPSHOT.exists():
        raise FileNotFoundError(f"Long-format snapshot not found: {LONG_SNAPSHOT}")

    df = pd.read_parquet(LONG_SNAPSHOT)
    logger.info(f"Loaded {len(df)} long-format rows for training from {LONG_SNAPSHOT}.")
    return df


# ---------------------------------------------------------
# Build features
# ---------------------------------------------------------


def _build_features(df_long: pd.DataFrame) -> pd.DataFrame:
    fb = FeatureBuilder()
    features_df = fb.build(df_long)

    if features_df.empty:
        raise ValueError("FeatureBuilder produced an empty DataFrame.")

    # Add synthetic status column (all nba_api games are completed)
    features_df["status"] = "final"

    return features_df


# ---------------------------------------------------------
# Train model
# ---------------------------------------------------------


def _train_model(features_df: pd.DataFrame) -> dict:
    """
    Train a simple RandomForest model.
    """

    # Remove the old broken filter:
    # train_df = features_df[features_df["status"] == "final"]
    # Instead, use all rows:
    train_df = features_df.copy()

    # Target: did the team win?
    y = train_df["won"]

    # Features
    feature_cols = [
        "is_home",
        "game_number",
        "rolling_points_for",
        "rolling_points_against",
        "rolling_win_rate",
        "lag_points_for",
        "lag_points_against",
        "lag_won",
        "opp_avg_points_for",
        "opp_avg_points_against",
        "opp_win_rate",
    ]

    X = train_df[feature_cols]

    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )

    logger.info("Training model...")
    model.fit(X_train, y_train)

    val_acc = model.score(X_val, y_val)
    logger.info(f"Validation accuracy: {val_acc:.4f}")

    return {
        "model": model,
        "val_acc": float(val_acc),
        "feature_cols": feature_cols,
    }


# ---------------------------------------------------------
# Save model to registry
# ---------------------------------------------------------


def _save_model(model_info: dict) -> dict:
    MODEL_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)

    model_id = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat()

    model_path = MODEL_REGISTRY_DIR / f"{model_id}.pkl"
    registry_path = MODEL_REGISTRY_DIR / "index.json"

    # Save model
    pd.to_pickle(model_info["model"], model_path)

    # Update registry
    entry = {
        "model_id": model_id,
        "timestamp": timestamp,
        "val_acc": model_info["val_acc"],
        "feature_cols": model_info["feature_cols"],
        "path": str(model_path),
    }

    if registry_path.exists():
        registry = json.loads(registry_path.read_text())
    else:
        registry = {"models": []}

    registry["models"].append(entry)
    registry_path.write_text(json.dumps(registry, indent=2))

    logger.success(f"Model saved → {model_path}")
    logger.success(f"Registry updated → {registry_path}")

    return entry


# ---------------------------------------------------------
# Main entry point
# ---------------------------------------------------------


def train_model() -> dict:
    logger.info("=== Model Training Start ===")

    df_long = _load_long_data()
    features_df = _build_features(df_long)
    model_info = _train_model(features_df)
    registry_entry = _save_model(model_info)

    logger.success("=== Model Training Complete ===")
    return registry_entry


if __name__ == "__main__":
    train_model()
