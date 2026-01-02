from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Predict Live Games
# File: src/scripts/predict_live.py
# Author: Sadiq
# ============================================================

from datetime import datetime
import pandas as pd
from loguru import logger

from src.ingestion.collector import fetch_scoreboard_for_date
from src.ingestion.normalizer.scoreboard_normalizer import normalize_scoreboard_to_wide
from src.ingestion.normalizer.wide_to_long import wide_to_long
from src.ingestion.normalizer.canonicalizer import canonicalize_team_game_df

from src.features.builder import FeatureBuilder
from src.pipeline.run_predictions import run_predictions

from src.config.paths import PREDICTIONS_DIR


def run_live_predictions() -> dict:
    logger.info("=== 🔴 Predicting Live Games (Canonical Pipeline) ===")

    today = datetime.utcnow().date()

    # --------------------------------------------------------
    # 1. Fetch live scoreboard
    # --------------------------------------------------------
    raw = fetch_scoreboard_for_date(today)
    if raw.empty:
        msg = "No live scoreboard data available."
        logger.warning(msg)
        return {"ok": False, "error": msg}

    # --------------------------------------------------------
    # 2. Normalize → wide → long → canonical
    # --------------------------------------------------------
    try:
        wide = normalize_scoreboard_to_wide(raw)
        long = wide_to_long(wide)
        long = canonicalize_team_game_df(long)
    except Exception as e:
        msg = f"Failed to normalize live scoreboard: {e}"
        logger.error(msg)
        return {"ok": False, "error": msg}

    if long.empty:
        msg = "Live scoreboard normalization produced no rows."
        logger.error(msg)
        return {"ok": False, "error": msg}

    # --------------------------------------------------------
    # 3. Build features (in-memory only)
    # --------------------------------------------------------
    try:
        fb = FeatureBuilder()  # version-agnostic
        features = fb.build(long)
    except Exception as e:
        msg = f"Feature building failed for live games: {e}"
        logger.error(msg)
        return {"ok": False, "error": msg}

    if features.empty:
        msg = "Could not build features for live games."
        logger.error(msg)
        return {"ok": False, "error": msg}

    # --------------------------------------------------------
    # 4. Predict win probabilities
    # --------------------------------------------------------
    try:
        preds = run_predictions(features)
    except Exception as e:
        msg = f"Prediction failed: {e}"
        logger.error(msg)
        return {"ok": False, "error": msg}

    preds["timestamp_utc"] = datetime.utcnow().isoformat()

    # --------------------------------------------------------
    # 5. Save predictions
    # --------------------------------------------------------
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PREDICTIONS_DIR / f"live_predictions_{today}.parquet"

    try:
        preds.to_parquet(out_path, index=False)
        logger.success(f"Live predictions saved to {out_path}")
    except Exception as e:
        msg = f"Failed to save live predictions: {e}"
        logger.error(msg)
        return {"ok": False, "error": msg}

    # --------------------------------------------------------
    # Human-readable summary
    # --------------------------------------------------------
    print("\n--- LIVE WIN PROBABILITIES ---")
    cols = ["team", "opponent", "points", "opponent_points", "win_probability"]
    print(preds[cols].to_string(index=False))
    print("\n=== DONE ===")

    return {
        "ok": True,
        "rows": len(preds),
        "output_path": str(out_path),
    }


def main():
    run_live_predictions()


if __name__ == "__main__":
    main()
