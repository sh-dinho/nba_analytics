from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v4
# Module: End-to-End Pipeline
# File: src/pipeline/run_end_to_end.py
# Author: Sadiq
#
# Description:
#     Full v4 pipeline:
#       1. Ingestion (yesterday + today)
#       2. Load canonical long snapshot
#       3. Build v4 features
#       4. Train moneyline model (classification)
#       5. Train totals model (regression)
#       6. Train spread models (regression + ATS classification)
#       7. Log summary
# ============================================================

from datetime import date, timedelta
import pandas as pd
from loguru import logger

from src.ingestion.pipeline import run_today_ingestion
from src.config.paths import LONG_SNAPSHOT
from src.features.builder import FeatureBuilder

# v4 training modules
from src.model.training_core import train_and_register_model
from src.model.train_totals import train_totals_model
from src.model.train_spread import train_spread_models


# ------------------------------------------------------------
# Main Orchestrator
# ------------------------------------------------------------


def run_end_to_end(pred_date: date | None = None) -> None:
    logger.info("🏀 Starting v4 End-to-End Pipeline")

    pred_date = pred_date or date.today()
    yesterday = pred_date - timedelta(days=1)

    # --------------------------------------------------------
    # Step 1 — Ingestion
    # --------------------------------------------------------
    logger.info("📥 Step 1: Ingestion (yesterday + today)")
    run_today_ingestion([yesterday, pred_date])

    # --------------------------------------------------------
    # Step 2 — Load canonical long snapshot
    # --------------------------------------------------------
    logger.info("🧱 Step 2: Loading Canonical Long Snapshot")

    if not LONG_SNAPSHOT.exists():
        raise FileNotFoundError(f"Long snapshot not found at {LONG_SNAPSHOT}")

    df = pd.read_parquet(LONG_SNAPSHOT)
    last_date = df["date"].max()

    logger.info(f"✅ Loaded canonical data: rows={len(df):,}, last_date={last_date}")

    # --------------------------------------------------------
    # Step 3 — Feature Building
    # --------------------------------------------------------
    logger.info("🧱 Step 3: Feature Building")

    fb = FeatureBuilder(version="v4")
    features_df = fb.build_from_long(df)

    logger.success(
        f"✅ Features built: rows={len(features_df):,}, cols={features_df.shape[1]}"
    )

    # --------------------------------------------------------
    # Step 4 — Training Models
    # --------------------------------------------------------
    logger.info("🧠 Step 4: Training Models")

    # -----------------------------
    # Moneyline (classification)
    # -----------------------------
    logger.info("Training model: moneyline")
    train_and_register_model(
        model_type="moneyline",
        df=features_df,
        feature_version="v4",
    )

    # -----------------------------
    # Totals (regression)
    # -----------------------------
    logger.info("Training model: totals")
    train_totals_model()

    # -----------------------------
    # Spread (regression + ATS classification)
    # -----------------------------
    logger.info("Training model: spread")
    train_spread_models()

    logger.success("🏁 End-to-End Pipeline Complete")


# ------------------------------------------------------------
# CLI Entrypoint
# ------------------------------------------------------------

if __name__ == "__main__":
    run_end_to_end()
