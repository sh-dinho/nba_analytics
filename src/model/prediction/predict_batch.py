from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Batch Prediction
# File: src/model/prediction/predict_batch.py
# Author: Sadiq
# ============================================================

import pandas as pd
from loguru import logger

from src.features.builder import FeatureBuilder
from src.model.registry import load_production_model
from src.model.prediction import (
    predict_moneyline,
    predict_totals,
    predict_spread,
    apply_threshold,
)


def predict_batch(input_path: str, output_path: str) -> None:
    logger.info(f"📦 Loading batch input from {input_path}")

    # --------------------------------------------------------
    # Load raw long-format input
    # --------------------------------------------------------
    df = pd.read_parquet(input_path)
    if df.empty:
        raise ValueError(f"Input file {input_path} is empty.")

    logger.info(f"Loaded {len(df)} raw rows")

    # --------------------------------------------------------
    # Build features (version-agnostic)
    # --------------------------------------------------------
    fb = FeatureBuilder()
    features = fb.build(df)

    if features.empty:
        raise ValueError("FeatureBuilder returned an empty feature set.")

    logger.info(f"Built features: {features.shape}")

    # --------------------------------------------------------
    # Load production models
    # --------------------------------------------------------
    ml_model, ml_meta = load_production_model("moneyline")
    totals_model, totals_meta = load_production_model("totals")
    spread_model, spread_meta = load_production_model("spread")

    logger.info(
        f"Loaded production models → "
        f"ML v{ml_meta.version}, Totals v{totals_meta.version}, Spread v{spread_meta.version}"
    )

    # --------------------------------------------------------
    # Run prediction heads
    # --------------------------------------------------------
    preds_ml = predict_moneyline(features, ml_model)
    preds_totals = predict_totals(features, totals_model)
    preds_spread = predict_spread(features, spread_model)

    # Drop model_type columns before merging
    preds_totals = preds_totals.drop(columns=["model_type"], errors="ignore")
    preds_spread = preds_spread.drop(columns=["model_type"], errors="ignore")

    # --------------------------------------------------------
    # Merge predictions
    # --------------------------------------------------------
    combined = (
        preds_ml
        .merge(preds_totals, on=["game_id", "team"], how="left")
        .merge(preds_spread, on=["game_id", "team"], how="left")
    )

    # Optional: apply win threshold
    combined = apply_threshold(combined)

    # --------------------------------------------------------
    # Save output
    # --------------------------------------------------------
    combined.to_parquet(output_path, index=False)
    logger.success(f"📤 Batch predictions saved → {output_path}")
