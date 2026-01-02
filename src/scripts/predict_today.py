from __future__ import annotations

from datetime import date
from loguru import logger

from src.ingestion.pipeline import ingest_single_date
from src.model.prediction.run_predictions import run_prediction_for_date
from src.config.paths import DATA_DIR


def run_predict_today() -> dict:
    logger.info("=== 🔮 Running Daily Prediction Job (v5 Canonical) ===")

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
    # 2. Run predictions (ML + totals + spread)
    # --------------------------------------------------------
    try:
        run_prediction_for_date(today)
    except Exception as e:
        msg = f"Prediction failed: {e}"
        logger.error(msg)
        return {"ok": False, "error": msg}

    # --------------------------------------------------------
    # 3. Summary
    # --------------------------------------------------------
    out_dir = DATA_DIR / "predictions"
    ml_path = out_dir / f"moneyline_{today}.parquet"

    logger.success(f"Predictions generated for {today}")
    return {
        "ok": True,
        "moneyline_path": str(ml_path),
        "totals_path": str(out_dir / f"totals_{today}.parquet"),
        "spread_path": str(out_dir / f"spread_{today}.parquet"),
    }


def main():
    run_predict_today()


if __name__ == "__main__":
    main()