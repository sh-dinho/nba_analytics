# File: scripts/generate_today_predictions.py

import os
import pandas as pd
import joblib
import argparse
import datetime
from scripts.utils import setup_logger

logger = setup_logger("generate_predictions")

DATA_DIR = "data"
MODELS_DIR = "models"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

PREDICTIONS_FILE = os.path.join(RESULTS_DIR, "today_predictions.csv")

def generate_today_predictions(threshold=0.6, strategy="kelly", max_fraction=0.05):
    model_file = os.path.join(MODELS_DIR, "game_predictor.pkl")
    features_file = os.path.join(DATA_DIR, "training_features.csv")

    if not os.path.exists(model_file):
        logger.error("No trained model found. Train a model first.")
        raise FileNotFoundError(model_file)
    if not os.path.exists(features_file):
        logger.error("No features file found. Build features first.")
        raise FileNotFoundError(features_file)

    logger.info("Loading model and features...")
    model = joblib.load(model_file)
    df = pd.read_csv(features_file)

    # ✅ Exclude identifiers and target
    drop_cols = ["home_win", "game_id", "home_team", "away_team"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].fillna(0)

    logger.info("Generating predictions...")
    preds = model.predict_proba(X)[:, 1]

    df["win_prob"] = preds
    df["bet_recommendation"] = (df["win_prob"] >= threshold).astype(int)

    if strategy == "kelly":
        df["bet_fraction"] = (df["win_prob"] - threshold) / (1 - threshold)
        df["bet_fraction"] = df["bet_fraction"].clip(lower=0, upper=max_fraction)
    else:
        df["bet_fraction"] = 0

    df.to_csv(PREDICTIONS_FILE, index=False)
    logger.info(f"Predictions saved to {PREDICTIONS_FILE}")

    # ✅ Summary line
    n_bets = int(df["bet_recommendation"].sum())
    avg_prob = df["win_prob"].mean()
    run_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary_line = (f"SUMMARY: Bets recommended={n_bets}, Avg win_prob={avg_prob:.3f}, "
                    f"Threshold={threshold}, Strategy={strategy}, MaxFraction={max_fraction}, RunTime={run_time}")
    logger.info(summary_line)

    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate today's predictions")
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--strategy", type=str, default="kelly")
    parser.add_argument("--max_fraction", type=float, default=0.05)
    args = parser.parse_args()

    generate_today_predictions(threshold=args.threshold,
                               strategy=args.strategy,
                               max_fraction=args.max_fraction)