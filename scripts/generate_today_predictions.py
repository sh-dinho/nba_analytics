# File: scripts/generate_today_predictions.py

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
logger = logging.getLogger("generate_predictions")


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

    # Assume all numeric columns except game_id, home_win are features for simplicity
    feature_cols = [c for c in features.select_dtypes(include="number").columns if c not in ["game_id", "home_win"]]
    X = features[feature_cols]

    preds = model.predict_proba(X)[:, 1]

    df = features[["game_id", "home_team", "away_team"]].copy()
    df["pred_home_win_prob"] = preds

    # Generate synthetic odds per game
    df["decimal_odds"] = generate_synthetic_odds(len(df))

    # Expected value calculation: EV = (P * O) - 1
    df["ev"] = df["pred_home_win_prob"] * df["decimal_odds"] - 1

    # Apply threshold filter
    df = df[df["pred_home_win_prob"] >= threshold]

    # 🌟 FIX: Use fixed filename 'predictions.csv'
    output_file = f"{RESULTS_DIR}/predictions.csv"

    df.to_csv(output_file, index=False)
    logger.info(f"✅ Predictions saved to {output_file} | Games predicted: {len(df)}")

    return df

if __name__ == "__main__":
    generate_today_predictions()