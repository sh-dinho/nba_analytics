# ============================================================
# File: src/daily_runner/daily_runner_mflow.py
# Purpose: Daily NBA prediction runner with MLflow logging + optional SHAP
# Version: 3.1 (uses unified nba_api_client)
# ============================================================

import logging
import os
from datetime import datetime
import random
import mlflow
import pandas as pd
from retrying import retry

from src.features.feature_engineering import generate_features_for_games
from src.prediction_engine.predictor import Predictor
from src.utils.mapping import map_team_ids
from mlflow_setup import start_run_with_metadata, configure_mlflow

# Unified NBA API client
from src.api.nba_api_client import fetch_today_games, update_historical_games

# Optional SHAP
SHAP_ENABLED = False
try:
    from src.interpretability.shap_analysis import run_shap
    SHAP_ENABLED = True
except ImportError:
    pass

# -----------------------------
# CONFIG
# -----------------------------
DATA_FILE = "data/cache/games_history.csv"
MODEL_PATH = "models/nba_logreg.pkl"
LOG_FILE = "data/logs/daily_runner.log"

os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------

def generate_predictions(today_games: pd.DataFrame, model_path: str) -> pd.DataFrame:
    if today_games.empty:
        return today_games
    features = generate_features_for_games(today_games.to_dict(orient="records"))
    features = map_team_ids(features, "TEAM_ID")
    predictor = Predictor(model_path=model_path)
    features["win_proba"] = predictor.predict_proba(features)
    features["win_pred"] = predictor.predict_label(features)
    # Confidence can be derived from win_proba instead of random
    features["prediction_confidence"] = features["win_proba"].apply(
        lambda x: x if x > 0.5 else 1 - x
    )
    return features

def log_to_mlflow(predictions: pd.DataFrame):
    if predictions.empty:
        return
    configure_mlflow()
    with start_run_with_metadata("daily_predictions"):
        mlflow.log_param("model_path", MODEL_PATH)
        csv_path = f"data/csv/daily_predictions_{datetime.now().strftime('%Y%m%d')}.csv"
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        predictions.to_csv(csv_path, index=False)
        mlflow.log_artifact(csv_path, artifact_path="daily_predictions")
        avg_proba = predictions["win_proba"].mean()
        mlflow.log_metric("avg_win_proba", avg_proba)
        logging.info(f"Logged predictions to MLflow. Avg win prob: {avg_proba:.2f}")

def print_summary(predictions: pd.DataFrame):
    if predictions.empty:
        logging.info("No NBA games today.")
        return
    logging.info("Today's NBA Predictions:")
    for _, row in predictions.iterrows():
        opponent_name = row.get("OPPONENT_TEAM_NAME", row["OPPONENT_TEAM_ID"])
        winner = row["TEAM_NAME"] if row["win_pred"] else opponent_name
        logging.info(
            f"{row['TEAM_NAME']} vs {opponent_name} | "
            f"Win probability: {row['win_proba']:.2f} | Predicted winner: {winner} | Confidence: {row['prediction_confidence']:.2f}"
        )

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Daily NBA Predictor Runner")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to trained model")
    parser.add_argument("--run_shap", action="store_true", help="Run SHAP analysis")
    args = parser.parse_args()

    logging.info("Starting MLflow NBA daily runner...")
    try:
        existing_df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        logging.info("No historical data found, starting fresh.")
        existing_df = pd.DataFrame(columns=["GAME_DATE","TEAM_NAME","MATCHUP","TEAM_ID","OPPONENT_TEAM_ID","PTS"])

    updated_games_df = update_historical_games(existing_df)
    today_games = fetch_today_games()
    predictions = generate_predictions(today_games, args.model)
    print_summary(predictions)
    log_to_mlflow(predictions)

    if args.run_shap and SHAP_ENABLED:
        shap_dir = "data/shap"
        os.makedirs(shap_dir, exist_ok=True)
        logging.info("Running SHAP analysis...")
        run_shap(args.model, cache_file=DATA_FILE, out_dir=shap_dir)

    logging.info("MLflow NBA daily runner finished.")
