# ============================================================
# File: src/daily_runner/daily_runner_mflow.py
# Purpose: Daily NBA prediction runner with MLflow logging + optional SHAP
# Project: nba_analysis
# Version: 3.4 (aligned schemas, dual team mapping, centralized logging)
# ============================================================

import logging
import os
from datetime import datetime
from typing import Optional

import mlflow
import numpy as np
import pandas as pd

from mlflow_setup import mlflow_run_context, log_system_metrics
from src.api.nba_api_client import (
    fetch_games,
    update_historical_games,
)  # use JSON client for IDs
from src.features.feature_engineering import generate_features_for_games
from src.prediction_engine.predictor import Predictor
from src.utils.mapping import map_team_ids

# -----------------------------
# CONFIG
# -----------------------------
DATA_FILE = "data/cache/games_history.csv"
MODEL_PATH = "models/nba_logreg.pkl"
LOG_FILE = "data/logs/daily_runner.log"

os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logger = logging.getLogger("daily_runner")
if not logger.handlers:
    fh = logging.FileHandler(LOG_FILE)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------


def _map_both_team_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add TEAM_NAME and OPPONENT_TEAM_NAME by applying team ID mapping to both columns.
    Assumes df has TEAM_ID and OPPONENT_TEAM_ID.
    """
    df = map_team_ids(df, team_col="TEAM_ID")
    df = df.rename(columns={"TEAM_NAME": "TEAM_NAME"})  # explicit for clarity

    # Map opponent separately; avoid overwriting TEAM_NAME
    df_opponent = df.rename(columns={"OPPONENT_TEAM_ID": "TEAM_ID"}).copy()
    df_opponent = map_team_ids(df_opponent, team_col="TEAM_ID")
    df["OPPONENT_TEAM_NAME"] = df_opponent["TEAM_NAME"]

    return df


def generate_predictions(today_games: pd.DataFrame, model_path: str) -> pd.DataFrame:
    if today_games is None or today_games.empty:
        return pd.DataFrame()

    # Ensure required columns exist
    required_cols = {"GAME_ID", "TEAM_ID", "OPPONENT_TEAM_ID", "GAME_DATE"}
    missing = required_cols - set(today_games.columns)
    if missing:
        logger.error("today_games missing required columns: %s", missing)
        return pd.DataFrame()

    # Feature generation expects a DataFrame
    features = generate_features_for_games(today_games)

    # Map team IDs to names for readability (both team and opponent)
    features = _map_both_team_names(features)

    predictor = Predictor(model_path=model_path)
    proba = predictor.predict_proba(
        features
    )  # array-like probabilities for TEAM_ID winning
    label = predictor.predict_label(features)  # boolean/int labels

    # Vectorized confidence: max(p, 1 - p)
    confidence = np.maximum(proba, 1.0 - proba)

    features["win_proba"] = proba
    features["win_pred"] = label
    features["prediction_confidence"] = confidence

    return features


def log_to_mlflow(predictions: pd.DataFrame, model_path: Optional[str] = None) -> None:
    if predictions is None or predictions.empty:
        return

    with mlflow_run_context("daily_predictions", strict=False):
        mlflow.log_param("model_path", model_path or MODEL_PATH)

        # Save CSV artifact
        csv_path = f"data/csv/daily_predictions_{datetime.now().strftime('%Y%m%d')}.csv"
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        predictions.to_csv(csv_path, index=False)
        mlflow.log_artifact(csv_path, artifact_path="daily_predictions")

        # Metrics
        avg_proba = float(np.nanmean(predictions["win_proba"]))
        mlflow.log_metric("avg_win_proba", avg_proba)
        mlflow.log_metric(
            "pred_confidence_mean",
            float(np.nanmean(predictions["prediction_confidence"])),
        )
        mlflow.log_metric(
            "pred_confidence_min",
            float(np.nanmin(predictions["prediction_confidence"])),
        )
        mlflow.log_metric(
            "pred_confidence_max",
            float(np.nanmax(predictions["prediction_confidence"])),
        )

        log_system_metrics()
        logger.info("Logged predictions to MLflow. Avg win prob: %.2f", avg_proba)


def print_summary(predictions: pd.DataFrame) -> None:
    if predictions is None or predictions.empty:
        logger.info("No NBA games today.")
        return

    logger.info("Today's NBA Predictions:")
    for _, row in predictions.iterrows():
        opponent_name = row.get("OPPONENT_TEAM_NAME", "Unknown")
        winner = row["TEAM_NAME"] if bool(row["win_pred"]) else opponent_name
        logger.info(
            "%s vs %s | Win probability: %.2f | Predicted winner: %s | Confidence: %.2f",
            row.get("TEAM_NAME", "Unknown"),
            opponent_name,
            float(row["win_proba"]),
            winner,
            float(row["prediction_confidence"]),
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

    logger.info("Starting MLflow NBA daily runner...")

    # Load existing historical data (used by update_historical_games)
    try:
        existing_df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        logger.info("No historical data found, starting fresh.")
        existing_df = pd.DataFrame(
            columns=[
                "GAME_ID",
                "GAME_DATE",
                "TEAM_NAME",
                "MATCHUP",
                "TEAM_ID",
                "OPPONENT_TEAM_ID",
                "PTS",
            ]
        )

    # Update historical games
    updated_games_df = update_historical_games(existing_df)

    # Fetch today's games using JSON client (ensures ID schema for features)
    today_str = datetime.now().strftime("%Y-%m-%d")
    today_games = fetch_games(today_str)

    # Predict
    predictions = generate_predictions(today_games, args.model)

    # Log and summarize
    log_system_metrics()  # snapshot after predictions
    print_summary(predictions)
    log_to_mlflow(predictions, model_path=args.model)

    # Optional SHAP analysis
    if args.run_shap:
        try:
            from src.interpretability.shap_analysis import run_shap

            shap_dir = "data/shap"
            os.makedirs(shap_dir, exist_ok=True)
            logger.info("Running SHAP analysis...")
            run_shap(args.model, cache_file=DATA_FILE, out_dir=shap_dir)
            log_system_metrics()  # snapshot after SHAP
        except ImportError:
            logger.warning("SHAP analysis requested, but shap module is not available.")

    logger.info("MLflow NBA daily runner finished.")
