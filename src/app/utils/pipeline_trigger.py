from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v5.0
# Name: Pipeline Trigger
# File: src/app/utils/pipeline_trigger.py
# Purpose:
#     Run ingestion + predictions end-to-end with:
#       • logging
#       • validation
#       • heartbeat writing
#       • optional backfill
#       • optional ingestion/prediction skipping
# ============================================================

from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional, List

from loguru import logger

from src.config.paths import DATA_DIR
from src.ingestion.pipeline import ingest_single_date
from src.model.prediction.run_predictions import run_prediction_for_date

# ------------------------------------------------------------
# Heartbeat Paths
# ------------------------------------------------------------
PIPELINE_HEARTBEAT = DATA_DIR / "pipeline_last_run.txt"
INGESTION_HEARTBEAT = DATA_DIR / "ingestion_last_run.txt"


# ------------------------------------------------------------
# Timestamp Helpers
# ------------------------------------------------------------
def _utc_timestamp() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


def _write_heartbeat(path: Path, label: str) -> None:
    ts = _utc_timestamp()
    try:
        path.write_text(ts)
        logger.info(f"[Pipeline] Wrote {label} heartbeat → {ts}")
    except Exception as e:
        logger.error(f"[Pipeline] Failed to write {label} heartbeat: {e}")


# ------------------------------------------------------------
# Prediction Validation
# ------------------------------------------------------------
def _validate_prediction_outputs(pred_date: date) -> None:
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


# ------------------------------------------------------------
# Prediction Runner
# ------------------------------------------------------------
def _run_predictions_for_date(pred_date: date) -> None:
    logger.info(f"🔮 Running predictions for {pred_date}")

    # v5 orchestrator handles ML + totals + spread
    run_prediction_for_date(pred_date)

    _validate_prediction_outputs(pred_date)


# ------------------------------------------------------------
# Full Pipeline Trigger
# ------------------------------------------------------------
def trigger_full_pipeline(
    pred_date: Optional[date] = None,
    backfill_days: int = 0,
    skip_ingestion: bool = False,
    skip_predictions: bool = False,
) -> str:
    try:
        logger.info("===============================================")
        logger.info("🚀 Starting Full Pipeline Run")
        logger.info("===============================================")

        # --------------------------------------------------------
        # 1. Ingestion
        # --------------------------------------------------------
        if not skip_ingestion:
            logger.info("📥 Running ingestion pipeline...")
            ingest_single_date(date.today())
            _write_heartbeat(INGESTION_HEARTBEAT, "ingestion")
        else:
            logger.info("⏭️ Skipping ingestion step (skip_ingestion=True)")

        # --------------------------------------------------------
        # 2. Predictions
        # --------------------------------------------------------
        if not skip_predictions:
            if backfill_days > 0:
                logger.info(f"📅 Backfilling predictions for last {backfill_days} days")

                today = date.today()
                dates_to_run: List[date] = [
                    today - timedelta(days=i) for i in range(backfill_days)
                ]

                for d in sorted(dates_to_run):
                    if d > today:
                        logger.warning(f"[Pipeline] Skipping future date: {d}")
                        continue
                    _run_predictions_for_date(d)

            else:
                target = pred_date or date.today()
                _run_predictions_for_date(target)

        else:
            logger.info("⏭️ Skipping prediction step (skip_predictions=True)")

        # --------------------------------------------------------
        # 3. Pipeline heartbeat
        # --------------------------------------------------------
        _write_heartbeat(PIPELINE_HEARTBEAT, "pipeline")

        logger.info("✅ Pipeline completed successfully.")
        return "Pipeline completed successfully."

    except Exception as e:
        logger.exception(f"❌ Pipeline failed: {e}")
        return f"Pipeline failed: {e}"