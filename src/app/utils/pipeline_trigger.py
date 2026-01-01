from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v5.0
# Name: Pipeline Trigger
# File: src/app/utils/pipeline_trigger.py
# Purpose: Run ingestion + predictions end-to-end with logging,
#          validation, and optional backfill support.
# ============================================================

from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional, List

from loguru import logger

from src.config.paths import DATA_DIR
from src.ingestion.pipeline import run_today_ingestion
from src.model.predict import run_prediction_for_date
from src.model.predict_totals import run_totals_prediction_for_date
from src.model.predict_spread import run_spread_prediction_for_date

PIPELINE_HEARTBEAT = DATA_DIR / "pipeline_last_run.txt"
INGESTION_HEARTBEAT = DATA_DIR / "ingestion_last_run.txt"


def _utc_timestamp() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


def _write_heartbeat(path: Path, label: str) -> None:
    ts = _utc_timestamp()
    path.write_text(ts)
    logger.info(f"[Pipeline] Wrote {label} heartbeat → {ts}")


def _validate_prediction_outputs(pred_date: date) -> None:
    """
    Ensure prediction files exist and are non-empty for the given date.
    Adjust file naming here to match your actual prediction outputs.
    """
    expected_files = [
        DATA_DIR / f"predictions/moneyline_{pred_date}.parquet",
        DATA_DIR / f"predictions/totals_{pred_date}.parquet",
        DATA_DIR / f"predictions/spread_{pred_date}.parquet",
    ]

    for f in expected_files:
        if not f.exists():
            logger.error(f"[Pipeline] Missing prediction file: {f}")
            raise FileNotFoundError(f"Missing prediction file: {f}")

        if f.stat().st_size == 0:
            logger.error(f"[Pipeline] Empty prediction file: {f}")
            raise ValueError(f"Empty prediction file: {f}")

    logger.info(f"[Pipeline] Validated prediction outputs for {pred_date}")


def trigger_full_pipeline(
    pred_date: Optional[date] = None,
    backfill_days: int = 0,
) -> str:
    """
    Runs ingestion + predictions end-to-end.
    Supports:
      - today's pipeline
      - backfill for N previous days based on backfill_days.
    """
    try:
        logger.info("===============================================")
        logger.info("🚀 Starting Full Pipeline Run")
        logger.info("===============================================")

        # 1. Ingestion
        logger.info("📥 Running ingestion pipeline...")
        run_today_ingestion()
        _write_heartbeat(INGESTION_HEARTBEAT, "ingestion")

        # 2. Predictions
        if backfill_days > 0:
            logger.info(f"📅 Backfilling predictions for last {backfill_days} days")
            today = date.today()
            dates_to_run: List[date] = [
                today - timedelta(days=i) for i in range(backfill_days)
            ]

            for d in dates_to_run:
                if d > today:
                    logger.warning(f"[Pipeline] Skipping future date: {d}")
                    continue

                logger.info(f"🔮 Running predictions for {d}")
                run_prediction_for_date(d)
                run_totals_prediction_for_date(d)
                run_spread_prediction_for_date(d)

                _validate_prediction_outputs(d)
        else:
            pred_date = pred_date or date.today()
            logger.info(f"🔮 Running predictions for {pred_date}")
            run_prediction_for_date(pred_date)
            run_totals_prediction_for_date(pred_date)
            run_spread_prediction_for_date(pred_date)
            _validate_prediction_outputs(pred_date)

        # 3. Pipeline heartbeat
        _write_heartbeat(PIPELINE_HEARTBEAT, "pipeline")

        logger.info("✅ Pipeline completed successfully.")
        return "Pipeline completed successfully."

    except Exception as e:
        logger.exception(f"❌ Pipeline failed: {e}")
        return f"Pipeline failed: {e}"
