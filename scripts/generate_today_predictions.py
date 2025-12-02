# File: scripts/generate_today_predictions.py
# FIX: Use config.py paths. Added Date column.

import os
import logging
import pandas as pd
import joblib
import datetime
from core.config import MODEL_FILE_PKL, PREDICTIONS_FILE, NEW_GAMES_FILE 
from scripts.build_features import build_features_for_new_games

logger = logging.getLogger(__name__)

def generate_today_predictions(threshold=0.6):
    """Generate predictions for today's games."""
    if not os.path.exists(MODEL_FILE_PKL):
        logger.error("No trained model found. Train a model first.")
        raise FileNotFoundError(MODEL_FILE_PKL)

    saved = joblib.load(MODEL_FILE_PKL)
    model = saved["model"]
    feature_order = saved["features"]
    logger.info(f"✅ Loaded model and feature order from {MODEL_FILE_PKL}")

    if not os.path.exists(NEW_GAMES_FILE):
        raise FileNotFoundError(f"{NEW_GAMES_FILE} not found. Run fetch_new_games.py first.")

    df = build_features_for_new_games(NEW_GAMES_FILE)

    X_num = df[feature_order]

    # TODO/WARNING: Proper imputation/scaling should use train-set scalers/imputers.
    # Leaving the original for basic robustness but strongly recommend fixing.
    X_num = X_num.fillna(0).replace([float("inf"), -float("inf")], 0)
    logger.warning("⚠️ Using simple fillna(0) for missing/inf values. Implement trained Imputer/Scaler for production.")

    # Predict probabilities
    probs = model.predict_proba(X_num)[:, 1]
    df["pred_home_win_prob"] = probs
    df["predicted_home_win"] = (probs >= threshold).astype(int)
    
    # CRITICAL FIX: Add today's date for deterministic bankroll tracking
    df['Date'] = datetime.datetime.now().strftime('%Y-%m-%d')
    
    os.makedirs(os.path.dirname(PREDICTIONS_FILE), exist_ok=True)
    df.to_csv(PREDICTIONS_FILE, index=False)
    logger.info(f"📊 Predictions saved to {PREDICTIONS_FILE}")

    return df

if __name__ == "__main__":
    generate_today_predictions()