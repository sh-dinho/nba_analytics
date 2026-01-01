from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Dataset Builder
# File: src/model/training/dataset_builder.py
# Author: Sadiq
#
# Description:
#     Build train/test datasets for a given model_type using
#     the canonical feature pipeline and model configuration.
# ============================================================

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from loguru import logger

from src.features.builder import FeatureBuilder
from src.model.config.model_config import (
    TRAINING_CONFIG,
    FEATURES,
    TARGET,
)
from src.config.paths import RAW_FEATURE_SNAPSHOT_PATH  # replace v5 snapshot


# ------------------------------------------------------------
# Dataset builder
# ------------------------------------------------------------

def build_dataset(model_type: str):
    """
    Build train/test sets for a given model_type.

    Args:
        model_type: "moneyline" | "totals" | "spread"

    Returns:
        X_train, X_test, y_train, y_test, feature_list
    """

    # --------------------------------------------------------
    # Load raw long-format snapshot
    # --------------------------------------------------------
    df = pd.read_parquet(RAW_FEATURE_SNAPSHOT_PATH)
    if df.empty:
        raise ValueError(f"Snapshot at {RAW_FEATURE_SNAPSHOT_PATH} is empty.")

    logger.info(f"Loaded snapshot: {df.shape}")

    # --------------------------------------------------------
    # Build canonical features
    # --------------------------------------------------------
    fb = FeatureBuilder()
    features = fb.build(df)

    if features.empty:
        raise ValueError("FeatureBuilder returned an empty feature set.")

    logger.info(f"Built features: {features.shape}")

    # --------------------------------------------------------
    # Select model-specific target
    # --------------------------------------------------------
    if model_type == "moneyline":
        target_col = TARGET  # "won"
    elif model_type == "totals":
        target_col = "total_points"
    elif model_type == "spread":
        target_col = "margin"
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' missing from snapshot.")

    y = df[target_col]

    # --------------------------------------------------------
    # Select feature matrix
    # --------------------------------------------------------
    missing = [f for f in FEATURES if f not in features.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    X = features[FEATURES]

    # --------------------------------------------------------
    # Train/test split
    # --------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TRAINING_CONFIG["test_size"],
        random_state=TRAINING_CONFIG["random_state"],
        shuffle=True,
    )

    logger.success(
        f"Dataset built → "
        f"X_train={X_train.shape}, X_test={X_test.shape}, "
        f"y_train={y_train.shape}, y_test={y_test.shape}"
    )

    return X_train, X_test, y_train, y_test, FEATURES