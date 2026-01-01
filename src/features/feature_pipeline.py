from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Feature Pipeline
# File: src/features/feature_pipeline.py
# Author: Sadiq
#
# Description:
#     End-to-end feature pipeline:
#       - takes canonical team-game rows (long_df)
#       - applies all feature modules in correct order
#       - validates each row against FeatureRow
#       - persists model-ready features to FEATURES_SNAPSHOT
# ============================================================

import pandas as pd
from loguru import logger

from src.config.paths import FEATURES_SNAPSHOT
from src.features.feature_schema import FeatureRow

# Feature modules
from src.features.elo import add_elo_features
from src.features.elo_rolling import add_elo_rolling_features
from src.features.rolling import add_rolling_features
from src.features.form import add_form_features
from src.features.rest import add_rest_features
from src.features.sos import add_sos_features
from src.features.opponent_adjusted import add_opponent_adjusted_features
from src.features.margin_features import add_margin_features


def _validate_feature_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Validate each row using FeatureRow."""
    errors = []
    validated = []

    for idx, row in df.iterrows():
        try:
            model = FeatureRow(**row.to_dict())
            validated.append(model.model_dump())
        except Exception as e:
            errors.append(f"Row {idx}: {e}")

    if errors:
        for e in errors[:20]:
            logger.error(e)
        raise ValueError(f"Feature validation failed for {len(errors)} rows.")

    return pd.DataFrame(validated)


def build_features(long_df: pd.DataFrame, persist: bool = False) -> pd.DataFrame:
    logger.info("Building features...")

    df = long_df.copy()

    # --------------------------------------------------------
    # Apply feature modules in correct order
    # --------------------------------------------------------
    df = add_elo_features(df)
    df = add_rolling_features(df)
    df = add_form_features(df)
    df = add_rest_features(df)
    df = add_sos_features(df)
    df = add_opponent_adjusted_features(df)
    df = add_margin_features(df)
    df = add_elo_rolling_features(df)

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------
    df = _validate_feature_rows(df)

    # --------------------------------------------------------
    # Persist
    # --------------------------------------------------------
    if persist:
        FEATURES_SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(FEATURES_SNAPSHOT, index=False)
        logger.success(f"Features persisted to {FEATURES_SNAPSHOT}")

    return df