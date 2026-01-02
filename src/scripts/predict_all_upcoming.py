from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Predict All Upcoming Games
# File: src/scripts/predict_all_upcoming.py
# Author: Sadiq
# ============================================================

from datetime import date
import pandas as pd
from loguru import logger

from src.config.paths import CANONICAL_SCHEDULE, PREDICTIONS_DIR
from src.ingestion.pipeline import ingest_single_date
from src.features.builder import FeatureBuilder
from src.pipeline.run_predictions import run_predictions


def run_predict_all_upcoming() -> dict:
    logger.info("=== Predicting All Upcoming Games ===")

    today = date.today()

    # --------------------------------------------------------
    # 1. Load canonical season schedule
    # --------------------------------------------------------
    if not CANONICAL_SCHEDULE.exists():
        msg = f"Canonical schedule not found at {CANONICAL_SCHEDULE}"
        logger.error(msg)
        return {"ok": False, "error": msg}

    try:
        schedule = pd.read_parquet(CANONICAL_SCHEDULE)
    except Exception as e:
        msg = f"Failed to read canonical schedule: {e}"
        logger.error(msg)
        return {"ok": False, "error": msg}

    schedule["date"] = pd.to_datetime(schedule["date"], errors="coerce").dt.date
    upcoming_dates = sorted({d for d in schedule["date"] if d and d >= today})

    if not upcoming_dates:
        logger.warning("No upcoming games found.")
        return {"ok": True, "predictions": 0, "message": "No upcoming games"}

    logger.info(f"Found {len(upcoming_dates)} upcoming dates.")

    fb = FeatureBuilder()  # version-agnostic
    all_predictions = []

    # --------------------------------------------------------
    # 2. Ingest + predict each date
    # --------------------------------------------------------
    for d in upcoming_dates:
        logger.info(f"Processing {d}")

        # Ingest
        try:
            long_df = ingest_single_date(d)
        except Exception as e:
            logger.error(f"Ingestion failed for {d}: {e}")
            continue

        if long_df.empty:
            logger.warning(f"No games ingested for {d}")
            continue

        # Features
        try:
            features_df = fb.build(long_df)
        except Exception as e:
            logger.error(f"Feature building failed for {d}: {e}")
            continue

        if features_df.empty:
            logger.warning(f"No features generated for {d}")
            continue

        # Predictions
        try:
            pred_df = run_predictions(features_df)
        except Exception as e:
            logger.error(f"Prediction failed for {d}: {e}")
            continue

        pred_df["prediction_date"] = d
        all_predictions.append(pred_df)

    if not all_predictions:
        logger.warning("No predictions generated.")
        return {"ok": False, "error": "No predictions generated"}

    final_df = pd.concat(all_predictions, ignore_index=True)

    # --------------------------------------------------------
    # 3. Save predictions
    # --------------------------------------------------------
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PREDICTIONS_DIR / f"predictions_upcoming_{today}.parquet"

    try:
        final_df.to_parquet(out_path, index=False)
        logger.success(f"Upcoming predictions saved to {out_path}")
    except Exception as e:
        msg = f"Failed to save predictions: {e}"
        logger.error(msg)
        return {"ok": False, "error": msg}

    # --------------------------------------------------------
    # Human-readable summary
    # --------------------------------------------------------
    print("\n=== UPCOMING GAME PREDICTIONS ===")
    print(
        final_df[
            ["game_id", "team", "opponent", "prediction_date", "win_probability"]
        ].head(20)
    )
    print("\n=== DONE ===")

    return {
        "ok": True,
        "predictions": len(final_df),
        "output_path": str(out_path),
    }


def main():
    run_predict_all_upcoming()


if __name__ == "__main__":
    main()
