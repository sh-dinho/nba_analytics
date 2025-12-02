# File: scripts/generate_predictions.py

import pandas as pd
import os
import numpy as np
import logging
import datetime
import joblib

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

def generate_synthetic_odds(n_games, low=1.5, high=3.0):
    """Generate synthetic decimal odds for each game."""
    return np.round(np.random.uniform(low, high, size=n_games), 2)

def generate_today_predictions(threshold=0.6):
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

    # Load features and model
    features = pd.read_csv(features_file)
    model = joblib.load(model_file)

    X = features.drop(columns=["game_id", "home_win"])
    preds = model.predict_proba(X)[:, 1]

    df = features[["game_id"]].copy()
    df["pred_home_win_prob"] = preds

    # Generate synthetic odds per game
    df["decimal_odds"] = generate_synthetic_odds(len(df))

    # Expected value calculation
    df["ev"] = df["pred_home_win_prob"] * df["decimal_odds"] - 1

    # Apply threshold filter
    df = df[df["pred_home_win_prob"] >= threshold]

    # Save with timestamped filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{RESULTS_DIR}/predictions_{timestamp}.csv"
    df.to_csv(output_file, index=False)

    logging.info(
        f"✅ Predictions saved to {output_file} | Games: {len(df)} | "
        f"Avg Prob: {df['pred_home_win_prob'].mean():.3f} | Avg EV: {df['ev'].mean():.3f}"
    )

    return df