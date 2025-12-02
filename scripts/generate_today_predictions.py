import pandas as pd
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

RESULTS_DIR = "results"
MODELS_DIR = "models"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

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
    preds = model.predict_proba(X)[:,1]

    df = features[["game_id"]].copy()
    df["pred_home_win_prob"] = preds
    df["decimal_odds"] = 1.9  # default odds, replace with real odds fetching logic
    df["ev"] = df["pred_home_win_prob"] * df["decimal_odds"] - 1
    df = df[df["pred_home_win_prob"] >= threshold]

    df.to_csv(f"{RESULTS_DIR}/predictions.csv", index=False)
    print(f"Predictions saved to {RESULTS_DIR}/predictions.csv")
    return df
