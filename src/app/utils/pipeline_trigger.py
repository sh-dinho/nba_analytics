from __future__ import annotations
from datetime import datetime, date

from src.config.paths import DATA_DIR
from src.ingestion.pipeline import run_today_ingestion
from src.model.predict import run_prediction_for_date
from src.model.predict_totals import run_totals_prediction_for_date
from src.model.predict_spread import run_spread_prediction_for_date

PIPELINE_HEARTBEAT = DATA_DIR / "pipeline_last_run.txt"
INGESTION_HEARTBEAT = DATA_DIR / "ingestion_last_run.txt"


def trigger_full_pipeline(pred_date: date | None = None) -> str:
    """
    Runs ingestion + predictions end-to-end using the REAL v4 pipeline.
    """
    try:
        # 1. INGESTION (yesterday + today)
        run_today_ingestion()
        INGESTION_HEARTBEAT.write_text(datetime.utcnow().isoformat())

        # 2. PREDICTIONS
        pred_date = pred_date or date.today()
        run_prediction_for_date(pred_date)
        run_totals_prediction_for_date(pred_date)
        run_spread_prediction_for_date(pred_date)

        PIPELINE_HEARTBEAT.write_text(datetime.utcnow().isoformat())

        return "Pipeline completed successfully."

    except Exception as e:
        return f"Pipeline failed: {e}"
