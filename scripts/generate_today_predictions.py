import os
import logging
import pandas as pd
import joblib
from config import MODEL_FILE, PREDICTIONS_FILE

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

def generate_today_predictions(threshold=0.6):
    if not os.path.exists(MODEL_FILE):
        logger.error("No trained model found. Train a model first.")
        raise FileNotFoundError(MODEL_FILE)

    model = joblib.load(MODEL_FILE)
    logger.info(f"✅ Loaded model from {MODEL_FILE}")

    # Example: load today's games
    df = pd.read_csv("data/new_games.csv")

    X_num = df.select_dtypes(include="number")
    X_num = X_num.fillna(0).replace([float("inf"), -float("inf")], 0)

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