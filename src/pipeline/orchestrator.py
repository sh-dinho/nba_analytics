from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics Orchestrator v3
# File: src/pipeline/orchestrator.py
# Author: Sadiq
#
# Description:
#     Single entry point for the daily NBA engine:
#       1. Ingest yesterday + today (ScoreboardV3)
#       2. Build canonical, long-format, and feature snapshots
#       3. Run prediction pipelines:
#            - Moneyline (win probability)
#            - Totals (over/under)
#            - Spread (ATS)
#       4. Log a clear summary and return structured results.
# ============================================================

from datetime import date
from typing import Optional, Dict, Any

from loguru import logger

from src.ingestion.pipeline import run_today_ingestion
from src.model.predict import run_prediction_for_date as run_moneyline
from src.model.predict_totals import run_prediction_for_date as run_totals
from src.model.predict_spread import run_prediction_for_date as run_spread


def run_daily_ingestion(pred_date: date) -> None:
    """
    Run ingestion for yesterday + today, rebuilding:
      - canonical schedule snapshot
      - long-format snapshot
      - feature snapshot (via FeatureBuilder)
    """
    logger.info(f"📥 Starting daily ingestion around prediction date {pred_date}")
    _ = run_today_ingestion(today=pred_date, feature_version="v1")
    logger.success("📥 Daily ingestion complete (canonical + long + features updated).")


def run_prediction_stage(pred_date: date) -> Dict[str, Optional[Any]]:
    """
    Run all prediction pipelines for the given date.
    Returns a dict with DataFrames (or None on failure) for:
      - moneyline
      - totals
      - spread
    """
    logger.info(f"🤖 Starting prediction stage for {pred_date}")

    results: Dict[str, Optional[Any]] = {
        "moneyline": None,
        "totals": None,
        "spread": None,
    }

    # --------------------------------------------------------
    # Moneyline
    # --------------------------------------------------------
    try:
        logger.info("🔹 Running Moneyline Predictions...")
        df_moneyline = run_moneyline(pred_date)
        results["moneyline"] = df_moneyline
        logger.info(f"   ✔ Moneyline predictions complete ({len(df_moneyline)} rows)")
    except Exception as e:
        logger.error(f"   ❌ Moneyline prediction failed: {e}")

    # --------------------------------------------------------
    # Totals
    # --------------------------------------------------------
    try:
        logger.info("🔹 Running Totals Predictions...")
        df_totals = run_totals(pred_date)
        results["totals"] = df_totals
        logger.info(f"   ✔ Totals predictions complete ({len(df_totals)} rows)")
    except Exception as e:
        logger.error(f"   ❌ Totals prediction failed: {e}")

    # --------------------------------------------------------
    # Spread
    # --------------------------------------------------------
    try:
        logger.info("🔹 Running Spread Predictions...")
        df_spread = run_spread(pred_date)
        results["spread"] = df_spread
        logger.info(f"   ✔ Spread predictions complete ({len(df_spread)} rows)")
    except Exception as e:
        logger.error(f"   ❌ Spread prediction failed: {e}")

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    logger.info("--------------------------------------------------")
    logger.info("🏁 Prediction Summary:")
    logger.info(
        f"   Moneyline: {'OK' if results['moneyline'] is not None else 'FAILED'}"
    )
    logger.info(f"   Totals:    {'OK' if results['totals'] is not None else 'FAILED'}")
    logger.info(f"   Spread:    {'OK' if results['spread'] is not None else 'FAILED'}")
    logger.info("--------------------------------------------------")

    return results


def run_full_pipeline(pred_date: Optional[date] = None) -> Dict[str, Optional[Any]]:
    """
    Full daily pipeline:
      1. Ingest (yesterday + today)
      2. Build features (via ingestion pipeline)
      3. Run all prediction heads for pred_date
    """
    pred_date = pred_date or date.today()
    logger.info(f"🏀 Starting FULL daily pipeline for {pred_date}")

    # 1) Ingestion / features
    run_daily_ingestion(pred_date)

    # 2) Predictions
    results = run_prediction_stage(pred_date)

    logger.success(f"🏀 Full daily pipeline complete for {pred_date}")
    return results


if __name__ == "__main__":
    run_full_pipeline()
