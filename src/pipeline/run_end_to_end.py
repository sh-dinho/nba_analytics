from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: End-to-End Pipeline Runner
# File: src/pipeline/run_end_to_end.py
# Author: Sadiq
#
# Description:
#     Modern end-to-end pipeline:
#       • load canonical long snapshot
#       • build features
#       • run predictions
#       • save predictions
#       • auto-retrain (optional)
#
#     This is the single entrypoint for daily automation.
# ============================================================

import argparse
from datetime import date
import pandas as pd
from loguru import logger

from src.features.builder import FeatureBuilder
from src.pipeline.run_predictions import run_predictions
from src.pipeline.auto_retrain import auto_retrain
from src.config.paths import (
    LONG_SNAPSHOT,
    PREDICTIONS_DIR,
)


def run_end_to_end(
    day: date,
    feature_version: str,
    model_version: str,
    run_retrain: bool = False,
):
    logger.info(f"🚀 Running end-to-end pipeline for {day}")

    # --------------------------------------------------------
    # 1. Load canonical long snapshot
    # --------------------------------------------------------
    logger.info("📥 Loading canonical long snapshot...")
    try:
        df_long = pd.read_parquet(LONG_SNAPSHOT)
    except Exception as e:
        logger.error(f"Failed to load LONG_SNAPSHOT: {e}")
        return pd.DataFrame()

    if df_long.empty:
        logger.error("LONG_SNAPSHOT is empty — aborting.")
        return pd.DataFrame()

    logger.info(f"Loaded long snapshot: {len(df_long)} rows")

    # --------------------------------------------------------
    # 2. Build features
    # --------------------------------------------------------
    logger.info("🏗️ Building features...")
    fb = FeatureBuilder(version=feature_version)

    try:
        features = fb.build(df_long)
    except Exception as e:
        logger.error(f"Feature building failed: {e}")
        return pd.DataFrame()

    if features.empty:
        logger.error("Feature builder returned empty DataFrame — aborting.")
        return pd.DataFrame()

    logger.info(f"Built features: {features.shape}")

    # --------------------------------------------------------
    # 3. Run predictions
    # --------------------------------------------------------
    logger.info("🔮 Running predictions...")
    preds = run_predictions(df_long, feature_version=feature_version)

    if preds.empty:
        logger.error("Prediction runner returned empty DataFrame — aborting.")
        return preds

    # --------------------------------------------------------
    # 4. Save predictions
    # --------------------------------------------------------
    out_path = PREDICTIONS_DIR / f"predictions_{day}.parquet"
    preds.to_parquet(out_path, index=False)
    logger.success(f"📤 Saved predictions → {out_path}")

    # --------------------------------------------------------
    # 5. Auto-retrain (optional)
    # --------------------------------------------------------
    if run_retrain:
        logger.info("🔄 Running auto-retrain...")
        auto_retrain(
            feature_version=feature_version,
            model_version=model_version,
        )

    logger.success("✨ End-to-end pipeline complete.")
    return preds


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run end-to-end pipeline")
    parser.add_argument("--date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--feature_version", required=True)
    parser.add_argument("--model_version", required=True)
    parser.add_argument("--retrain", action="store_true")

    args = parser.parse_args()
    day = date.fromisoformat(args.date)

    run_end_to_end(
        day,
        feature_version=args.feature_version,
        model_version=args.model_version,
        run_retrain=args.retrain,
    )