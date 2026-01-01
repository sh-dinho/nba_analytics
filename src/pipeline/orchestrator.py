from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Daily Orchestrator
# File: src/pipeline/orchestrator.py
# Author: Sadiq
#
# Description:
#     Daily automation pipeline:
#         1. Ingest raw data
#         2. Persist canonical long snapshot
#         3. Build features
#         4. Run predictions
#         5. Export dashboard files
#         6. Auto-retrain (optional)
# ============================================================

import pandas as pd
from loguru import logger

from src.ingestion.pipeline import ingest_single_date
from src.features.builder import FeatureBuilder
from src.pipeline.run_predictions import run_predictions
from src.pipeline.auto_retrain import auto_retrain
from src.config.paths import (
    LONG_SNAPSHOT,
    PREDICTIONS_DIR,
    DASHBOARD_DIR,
)


def orchestrate_daily(
    pred_date: str,
    feature_version: str,
    model_version: str,
    run_retrain: bool = False,
    export_dashboard: bool = True,
):
    logger.info(f"🧠 Starting daily orchestrator for {pred_date}")

    # --------------------------------------------------------
    # 1. Ingest raw data
    # --------------------------------------------------------
    logger.info("📥 Ingesting raw data...")
    new_rows = ingest_single_date(pred_date, feature_groups=None)

    if new_rows.empty:
        logger.warning("No new rows ingested — aborting.")
        return pd.DataFrame()

    # --------------------------------------------------------
    # 2. Persist canonical long snapshot
    # --------------------------------------------------------
    try:
        new_rows.to_parquet(LONG_SNAPSHOT, index=False)
        logger.success(f"📦 Saved canonical snapshot → {LONG_SNAPSHOT}")
    except Exception as e:
        logger.error(f"Failed to save canonical snapshot: {e}")
        return pd.DataFrame()

    # --------------------------------------------------------
    # 3. Build features
    # --------------------------------------------------------
    logger.info("🏗️ Building features...")
    fb = FeatureBuilder(version=feature_version)

    try:
        features = fb.build(new_rows)
    except Exception as e:
        logger.error(f"Feature building failed: {e}")
        return pd.DataFrame()

    if features.empty:
        logger.error("Feature builder returned empty DataFrame — aborting.")
        return pd.DataFrame()

    logger.info(f"Built features: {features.shape}")

    # --------------------------------------------------------
    # 4. Run predictions
    # --------------------------------------------------------
    logger.info("🔮 Running predictions...")
    preds = run_predictions(new_rows, feature_version=feature_version)

    if preds.empty:
        logger.error("Prediction runner returned empty DataFrame — aborting.")
        return preds

    # Save predictions
    pred_path = PREDICTIONS_DIR / f"predictions_{pred_date}.parquet"
    preds.to_parquet(pred_path, index=False)
    logger.success(f"📤 Saved predictions → {pred_path}")

    # --------------------------------------------------------
    # 5. Export dashboard files
    # --------------------------------------------------------
    if export_dashboard:
        DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
        dash_path = DASHBOARD_DIR / f"dashboard_{pred_date}.csv"
        preds.to_csv(dash_path, index=False)
        logger.success(f"📊 Dashboard export ready → {dash_path}")

    # --------------------------------------------------------
    # 6. Auto-retrain (optional)
    # --------------------------------------------------------
    if run_retrain:
        logger.info("🔄 Running auto-retrain...")
        auto_retrain(
            feature_version=feature_version,
            model_version=model_version,
        )

    logger.success("📣 Daily pipeline complete.")
    return preds