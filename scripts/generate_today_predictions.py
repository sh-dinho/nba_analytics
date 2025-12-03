# ============================================================
# File: scripts/generate_today_predictions.py
# Purpose: Generate predictions for today's games
# ============================================================

import os
import pandas as pd
import joblib
import datetime
from core.config import MODEL_FILE_PKL, PREDICTIONS_FILE, NEW_GAMES_FILE
from core.log_config import setup_logger
from core.exceptions import PipelineError, DataError
from core.utils import ensure_columns
from scripts.build_features import build_features_for_new_games

logger = setup_logger("generate_today_predictions")


def generate_today_predictions(threshold: float = 0.6) -> pd.DataFrame:
    """Generate predictions for today's games."""
    if not os.path.exists(MODEL_FILE_PKL):
        logger.error("❌ No trained model found. Train a model first.")
        raise FileNotFoundError(MODEL_FILE_PKL)

    try:
        saved = joblib.load(MODEL_FILE_PKL)
        model = saved["model"]
        feature_order = saved["features"]
        logger.info(f"✅ Loaded model and feature order from {MODEL_FILE_PKL}")
    except Exception as e:
        raise PipelineError(f"Failed to load model from {MODEL_FILE_PKL}: {e}")

    if not os.path.exists(NEW_GAMES_FILE):
        raise FileNotFoundError(f"{NEW_GAMES_FILE} not found. Run fetch_new_games.py first.")

    df = build_features_for_new_games(NEW_GAMES_FILE)

    # Validate required features
    try:
        ensure_columns(df, set(feature_order), "new game features")
    except ValueError as e:
        raise DataError(str(e))

    X_num = df[feature_order]

    # TODO/WARNING: Proper imputation/scaling should use train-set scalers/imputers.
    # Leaving the original for basic robustness but strongly recommend fixing.
    X_num = X_num.fillna(0).replace([float("inf"), -float("inf")], 0)
    logger.warning("⚠️ Using simple fillna(0) for missing/inf values. Implement trained Imputer/Scaler for production.")

    try:
        # Predict probabilities
        probs = model.predict_proba(X_num)[:, 1]
    except Exception as e:
        raise PipelineError(f"Prediction failed: {e}")

    df["pred_home_win_prob"] = probs
    df["predicted_home_win"] = (probs >= threshold).astype(int)

    # CRITICAL FIX: Add today's date for deterministic bankroll tracking
    df["Date"] = datetime.datetime.now().strftime("%Y-%m-%d")

    os.makedirs(os.path.dirname(PREDICTIONS_FILE), exist_ok=True)
    df.to_csv(PREDICTIONS_FILE, index=False)
    logger.info(f"📊 Predictions saved to {PREDICTIONS_FILE} ({len(df)} rows)")

    return df


if __name__ == "__main__":
    generate_today_predictions()