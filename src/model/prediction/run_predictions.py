from __future__ import annotations

import pandas as pd
from datetime import date
from pathlib import Path
from loguru import logger

from src.config.paths import DATA_DIR, LONG_SNAPSHOT
from src.features.builder import FeatureBuilder
from src.model.registry import load_production_model
from src.model.prediction import (
    predict_moneyline,
    predict_totals,
    predict_spread,
    apply_threshold,
)


def run_prediction_for_date(pred_date: date) -> None:
    logger.info(f"🔮 Running ML/Totals/Spread predictions for {pred_date}")

    # --------------------------------------------------------
    # Load canonical long snapshot
    # --------------------------------------------------------
    if not LONG_SNAPSHOT.exists():
        raise FileNotFoundError(f"Snapshot missing: {LONG_SNAPSHOT}")

    df = pd.read_parquet(LONG_SNAPSHOT)
    df = df[df["date"] == pd.Timestamp(pred_date)]

    if df.empty:
        logger.warning(f"No rows found for {pred_date} in snapshot.")
        return

    # --------------------------------------------------------
    # Build features
    # --------------------------------------------------------
    fb = FeatureBuilder()
    features = fb.build(df)

    # Validate identity columns early
    for col in ["game_id", "team"]:
        if col not in features.columns:
            raise ValueError(f"FeatureBuilder output missing required column: '{col}'")

    # --------------------------------------------------------
    # Load models
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

    preds_ml = apply_threshold(preds_ml)

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------
    out_dir = DATA_DIR / "predictions"
    out_dir.mkdir(exist_ok=True)

    preds_ml.to_parquet(out_dir / f"moneyline_{pred_date}.parquet", index=False)
    preds_totals.to_parquet(out_dir / f"totals_{pred_date}.parquet", index=False)
    preds_spread.to_parquet(out_dir / f"spread_{pred_date}.parquet", index=False)

    # --------------------------------------------------------
    # Optional: Save combined predictions
    # --------------------------------------------------------
    # combined = (
    #     preds_ml
    #     .merge(preds_totals, on=["game_id", "team"], how="left")
    #     .merge(preds_spread, on=["game_id", "team"], how="left")
    # )
    # combined.to_parquet(out_dir / f"combined_{pred_date}.parquet", index=False)

    logger.success(f"Predictions saved for {pred_date}")
