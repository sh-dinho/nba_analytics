from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Run Predictions
# File: src/pipeline/run_predictions.py
# Author: Sadiq
#
# Description:
#     Unified prediction pipeline:
#       • build features
#       • load production models
#       • run prediction heads
#       • merge outputs
# ============================================================

import pandas as pd
from loguru import logger

from src.features.builder import FeatureBuilder
from src.model.registry import load_production_model
from src.model.prediction.predict import (
    predict_moneyline,
    predict_totals,
    predict_spread,
)


def run_predictions(
    df_long: pd.DataFrame,
    feature_version: str,
) -> pd.DataFrame:
    """
    Run predictions on a long-format DataFrame.

    Parameters
    ----------
    df_long : pd.DataFrame
        Canonical long-format dataset.
    feature_version : str
        Feature builder version tag.

    Returns
    -------
    pd.DataFrame
        Combined predictions for moneyline, totals, and spread.
    """

    # --------------------------------------------------------
    # 1. Build features
    # --------------------------------------------------------
    logger.info("🏗️ Building features...")
    fb = FeatureBuilder(version=feature_version)
    features = fb.build(df_long)

    if features.empty:
        logger.error("Feature builder returned empty DataFrame.")
        return pd.DataFrame()

    # --------------------------------------------------------
    # 2. Load production models
    # --------------------------------------------------------
    logger.info("📦 Loading production models...")
    ml_model, _ = load_production_model("moneyline")
    totals_model, _ = load_production_model("totals")
    spread_model, _ = load_production_model("spread")

    # --------------------------------------------------------
    # 3. Run prediction heads
    # --------------------------------------------------------
    logger.info("🔮 Running prediction heads...")
    preds_ml = predict_moneyline(features, ml_model)
    preds_totals = predict_totals(features, totals_model)
    preds_spread = predict_spread(features, spread_model)

    # --------------------------------------------------------
    # 4. Merge predictions
    # --------------------------------------------------------
    logger.info("🔗 Merging predictions...")
    combined = (
        preds_ml
        .merge(preds_totals, on=["game_id", "team"], how="left")
        .merge(preds_spread, on=["game_id", "team"], how="left")
    )

    logger.success(f"✨ Predictions ready: {len(combined)} rows")
    return combined