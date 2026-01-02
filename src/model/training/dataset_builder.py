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
#     Includes:
#       • season-aware splitting
#       • feature validation
#       • NaN-safe filtering
#       • metadata for model registry
# ============================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from loguru import logger

from src.config.paths import FEATURES_SNAPSHOT
from src.model.config.model_config import FEATURE_MAP, TARGET_MAP


def build_dataset(model_type: str):
    """
    Build train/test splits for a given model type.

    Returns:
        X_train, X_test, y_train, y_test, feature_list, metadata
    """
    logger.info(f"📦 Building dataset for model_type='{model_type}'")

    # Load snapshot
    df = pd.read_parquet(FEATURES_SNAPSHOT)
    if df.empty:
        raise RuntimeError("Feature snapshot is empty — cannot build dataset.")

    logger.info(f"Loaded {len(df):,} rows with {len(df.columns)} columns.")

    # Validate target
    target_col = TARGET_MAP[model_type]
    df = df.dropna(subset=[target_col])

    # Feature list
    feature_list = FEATURE_MAP[model_type]
    df = df.dropna(subset=feature_list)

    X = df[feature_list].copy()
    y = df[target_col].copy()

    # Season-aware split
    if "season" in df.columns:
        latest_season = df["season"].max()
        logger.info(f"Using season-aware split. Test season = {latest_season}")

        train_df = df[df["season"] < latest_season]
        test_df = df[df["season"] == latest_season]

        if not train_df.empty and not test_df.empty:
            X_train = train_df[feature_list]
            y_train = train_df[target_col]
            X_test = test_df[feature_list]
            y_test = test_df[target_col]

            metadata = {
                "train_start_date": train_df["date"].min(),
                "train_end_date": train_df["date"].max(),
                "test_start_date": test_df["date"].min(),
                "test_end_date": test_df["date"].max(),
            }

            logger.info(
                f"Dataset built (season split): "
                f"{len(X_train)} train rows, {len(X_test)} test rows"
            )
            return X_train, X_test, y_train, y_test, feature_list, metadata

    # Fallback random split
    stratify = y if model_type == "moneyline" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=stratify
    )

    metadata = {
        "train_start_date": df["date"].min(),
        "train_end_date": df["date"].max(),
        "test_start_date": None,
        "test_end_date": None,
    }

    logger.info(
        f"Dataset built (random split): {len(X_train)} train rows, "
        f"{len(X_test)} test rows"
    )

    return X_train, X_test, y_train, y_test, feature_list, metadata
