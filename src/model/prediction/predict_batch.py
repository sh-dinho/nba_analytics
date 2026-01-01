from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Batch Prediction
# File: src/model/prediction/predict_batch.py
# Author: Sadiq
#
# Description:
#     Run predictions on an arbitrary parquet file containing
#     long-format team-game rows. This bypasses the daily
#     ingestion pipeline and is useful for:
#       - backfills
#       - historical prediction generation
#       - ad-hoc experiments
# ============================================================

import pandas as pd
from loguru import logger

from src.features.builder import FeatureBuilder
from src.model.registry.registry import load_production_model
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

    # --------------------------------------------------------
    # Build features (version-agnostic)
    # --------------------------------------------------------
    fb = FeatureBuilder()  # no more version="v5"
    features = fb.build(df)

    if features.empty:
        raise ValueError("FeatureBuilder returned an empty feature set.")

    logger.info(f"Built features: {features.shape}")

    # --------------------------------------------------------
    # Load production models
    # --------------------------------------------------------
    ml_model, _ = load_production_model("moneyline")
    totals_model, _ = load_production_model("totals")
    spread_model, _ = load_production_model("spread")

    # --------------------------------------------------------
    # Run prediction heads
    # --------------------------------------------------------
    preds_ml = predict_moneyline(features, ml_model)
    preds_totals = predict_totals(features, totals_model)
    preds_spread = predict_spread(features, spread_model)

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