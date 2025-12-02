# File: scripts/generate_predictions.py

import pandas as pd
import os
import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier
import joblib
import datetime

RESULTS_DIR = "results"
MODELS_DIR = "models"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

def generate_today_predictions(threshold=0.6, odds=1.9):
    """
    Generate predictions for today's games.
    Returns a DataFrame with columns: game_id, pred_home_win_prob, decimal_odds, ev
    """
    features_file = "data/training_features.csv"
    model_file = f"{MODELS_DIR}/game_predictor.pkl"

    if not os.path.exists(features_file):
        raise FileNotFoundError(f"{features_file} not found. Generate training features first.")

    if not os.path.exists(model_file):
        raise FileNotFoundError(f"{model_file} not found. Train the model first.")

    try:
        features = pd.read_csv(features_file)
        model = joblib.load(model_file)
    except Exception as e:
        logging.error(f"Error loading files: {e}")
        raise

    # Prepare features
    X = features.drop(columns=["game_id", "home_win"])
    preds = model.predict_proba(X)[:, 1]

    # Build results DataFrame
    df = features[["game_id"]].copy()
    df["pred_home_win_prob"] = preds
    df["decimal_odds"] = odds
    df["ev"] = df["pred_home_win_prob"] * df["decimal_odds"] - 1
    df = df[df["pred_home_win_prob"] >= threshold]

    # Save with timestamped filename for versioning
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{RESULTS_DIR}/predictions_{timestamp}.csv"

    # Add header row with file path + name
    header_note = pd.DataFrame({"info": [f"File: {output_file}"]})
    df_out = pd.concat([header_note, df], axis=0)

    df_out.to_csv(output_file, index=False)

    # Summary log line
    avg_prob = df["pred_home_win_prob"].mean() if not df.empty else 0
    avg_ev = df["ev"].mean() if not df.empty else 0
    logging.info(
        f"✅ Predictions saved to {output_file} | Games: {len(df)} | "
        f"Avg Prob: {avg_prob:.3f} | Avg EV: {avg_ev:.3f}"
    )

    return df