import os
import logging
import pandas as pd
import joblib
from config import MODEL_FILE, PREDICTIONS_FILE
from scripts.build_features import build_features_for_new_games

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

def generate_today_predictions(threshold=0.6):
    """
    Generate predictions for today's games using the trained model.
    Ensures features match training by reordering columns to saved feature order.
    """
    if not os.path.exists(MODEL_FILE):
        logger.error("No trained model found. Train a model first.")
        raise FileNotFoundError(MODEL_FILE)

    saved = joblib.load(MODEL_FILE)
    model = saved["model"]
    feature_order = saved["features"]
    logger.info(f"✅ Loaded model and feature order from {MODEL_FILE}")

    if not os.path.exists("data/new_games.csv"):
        raise FileNotFoundError("data/new_games.csv not found. Run fetch_new_games.py first.")

    df = build_features_for_new_games("data/new_games.csv")

    # Reorder columns to match training
    X_num = df[feature_order].fillna(0).replace([float("inf"), -float("inf")], 0)

    # Predict probabilities
    probs = model.predict_proba(X_num)[:, 1]
    df["pred_home_win_prob"] = probs
    df["predicted_home_win"] = (probs >= threshold).astype(int)

    os.makedirs(os.path.dirname(PREDICTIONS_FILE), exist_ok=True)
    df.to_csv(PREDICTIONS_FILE, index=False)
    logger.info(f"📊 Predictions saved to {PREDICTIONS_FILE}")

    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.6)
    args = parser.parse_args()
    generate_today_predictions(threshold=args.threshold)