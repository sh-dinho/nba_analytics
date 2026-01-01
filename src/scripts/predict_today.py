from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Predict Today’s Games (Canonical Pipeline)
# File: src/scripts/predict_today.py
# Author: Sadiq
#
# Description:
#     Daily prediction job using the modern ingestion + feature
#     + prediction pipeline.
#
#     Steps:
#       • Ingest today's games (canonical ingestion)
#       • Build features (FeatureBuilder)
#       • Load latest model
#       • Predict win probabilities
#       • Save predictions to PREDICTIONS_DIR
#
#     Safe for cron, Airflow, GitHub Actions, or systemd timers.
# ============================================================

from datetime import date
import pandas as pd
from loguru import logger

from src.ingestion.pipeline import ingest_single_date
from src.features.builder import FeatureBuilder
from src.pipeline.run_predictions import run_predictions
from src.model.registry.load_model import load_latest_model
from src.config.paths import PREDICTIONS_DIR


def run_predict_today(feature_version: str = "v1") -> dict:
    """
    Predict win probabilities for today's games using the
    canonical ingestion + feature + prediction pipeline.
    """
    logger.info("=== 🔮 Running Daily Prediction Job (Canonical) ===")

    today = date.today()

    # --------------------------------------------------------
    # 1. Ingest today's games
    # --------------------------------------------------------
    try:
        long_df = ingest_single_date(today)
    except Exception as e:
        msg = f"Ingestion failed for {today}: {e}"
        logger.error(msg)
        return {"ok": False, "error": msg}

    if long_df.empty:
        msg = f"No games found for {today}."
        logger.warning(msg)
        return {"ok": True, "predictions": 0, "message": msg}

    # --------------------------------------------------------
    # 2. Build features
    # --------------------------------------------------------
    try:
        fb = FeatureBuilder(version=feature_version)
        features_df = fb.build(long_df)
    except Exception as e:
        msg = f"Feature building failed: {e}"
        logger.error(msg)
        return {"ok": False, "error": msg}

    if features_df.empty:
        msg = "Feature builder returned empty DataFrame."
        logger.error(msg)
        return {"ok": False, "error": msg}

    # --------------------------------------------------------
    # 3. Load latest model
    # --------------------------------------------------------
    try:
        model = load_latest_model()
    except Exception as e:
        msg = f"Failed to load latest model: {e}"
        logger.error(msg)
        return {"ok": False, "error": msg}

    # --------------------------------------------------------
    # 4. Predict win probabilities
    # --------------------------------------------------------
    try:
        pred_df = run_predictions(long_df, feature_version=feature_version)
    except Exception as e:
        msg = f"Prediction failed: {e}"
        logger.error(msg)
        return {"ok": False, "error": msg}

    # --------------------------------------------------------
    # 5. Save predictions
    # --------------------------------------------------------
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PREDICTIONS_DIR / f"predictions_{today}.parquet"

    try:
        pred_df.to_parquet(out_path, index=False)
        logger.success(f"Generated {len(pred_df)} predictions at {out_path}")
    except Exception as e:
        msg = f"Failed to save predictions: {e}"
        logger.error(msg)
        return {"ok": False, "error": msg}

    # --------------------------------------------------------
    # Human-readable summary
    # --------------------------------------------------------
    print("\n=== TODAY'S WIN PROBABILITIES ===")
    print(
        pred_df[["game_id", "team", "opponent", "win_probability"]]
        .sort_values("game_id")
        .to_string(index=False)
    )
    print("\n=== DONE ===")

    return {
        "ok": True,
        "predictions": len(pred_df),
        "output_path": str(out_path),
    }


def main():
    run_predict_today()


if __name__ == "__main__":
    main()