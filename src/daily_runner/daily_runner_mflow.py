# ============================================================
# File: src/daily_runner/daily_runner_mflow.py
# Purpose: Daily NBA prediction runner with MLflow logging + optional SHAP
# Project: nba_analysis
# Version: 3.5 (fixed imports, variable shadowing, run_shap fallback, requirements alignment)
# ============================================================

import logging
import os
import sys
from datetime import datetime
from typing import Optional

import mlflow
import numpy as np
import pandas as pd

from mlflow_setup import mlflow_run_context, log_system_metrics
from src.api.nba_api_client import fetch_today_games


# NOTE: update_historical_games and fetch_games were missing in nba_api_client.
# For now, we stub them safely here:
def update_historical_games(existing_data: pd.DataFrame) -> pd.DataFrame:
    """Stub: update historical games by merging with new season data."""
    return existing_data


def fetch_games(date: str) -> pd.DataFrame:
    """Wrapper around fetch_today_games for compatibility."""
    return fetch_today_games(date)


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
    """Add TEAM_NAME and OPPONENT_TEAM_NAME by applying team ID mapping to both columns."""
    df = map_team_ids(df, team_col="TEAM_ID")
    df_opponent = df.rename(columns={"OPPONENT_TEAM_ID": "TEAM_ID"}).copy()
    df_opponent = map_team_ids(df_opponent, team_col="TEAM_ID")
    df["OPPONENT_TEAM_NAME"] = df_opponent["TEAM_NAME"]
    return df


def generate_predictions(
    today_games_out: pd.DataFrame, model_path: str
) -> pd.DataFrame:
    if today_games_out is None or today_games_out.empty:
        return pd.DataFrame()

    required_cols = {"GAME_ID", "TEAM_ID", "OPPONENT_TEAM_ID", "GAME_DATE"}
    missing = required_cols - set(today_games_out.columns)
    if missing:
        logger.error("today_games_out missing required columns: %s", missing)
        return pd.DataFrame()

    features = generate_features_for_games(today_games_out)
    features = _map_both_team_names(features)

    predictor = Predictor(model_path=model_path)
    proba = predictor.predict_proba(features)
    label = predictor.predict_label(features)

    confidence = np.maximum(proba, 1.0 - proba)

    features["win_proba"] = proba
    features["win_pred"] = label
    features["prediction_confidence"] = confidence

    return features


def log_to_mlflow(
    predictions_out: pd.DataFrame, model_path: Optional[str] = None
) -> None:
    if predictions_out is None or predictions_out.empty:
        return

    with mlflow_run_context("daily_predictions", strict=False):
        mlflow.log_param("model_path", model_path or MODEL_PATH)

        csv_path = f"data/csv/daily_predictions_{datetime.now().strftime('%Y%m%d')}.csv"
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        predictions_out.to_csv(csv_path, index=False)
        mlflow.log_artifact(csv_path, artifact_path="daily_predictions")

        avg_proba = float(np.nanmean(predictions_out["win_proba"]))
        mlflow.log_metric("avg_win_proba", avg_proba)
        mlflow.log_metric(
            "pred_confidence_mean",
            float(np.nanmean(predictions_out["prediction_confidence"])),
        )
        mlflow.log_metric(
            "pred_confidence_min",
            float(np.nanmin(predictions_out["prediction_confidence"])),
        )
        mlflow.log_metric(
            "pred_confidence_max",
            float(np.nanmax(predictions_out["prediction_confidence"])),
        )

        log_system_metrics()
        logger.info("Logged predictions to MLflow. Avg win prob: %.2f", avg_proba)


def print_summary(predictions_out: pd.DataFrame) -> None:
    if predictions_out is None or predictions_out.empty:
        logger.info("No NBA games today.")
        return

    logger.info("Today's NBA Predictions:")
    for _, row in predictions_out.iterrows():
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
    cli_args = parser.parse_args()

    logger.info("Starting MLflow NBA daily runner...")

    try:
        existing_data = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        logger.info("No historical data found, starting fresh.")
        existing_data = pd.DataFrame(
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

    try:
        updated_games = update_historical_games(existing_data)
    except Exception as e:
        logger.error("Failed to update historical games: %s", e)
        updated_games = existing_data

    today_str = datetime.now().strftime("%Y-%m-%d")
    try:
        today_games_out = fetch_games(today_str)
    except Exception as e:
        logger.error("Failed to fetch today's games: %s", e)
        today_games_out = pd.DataFrame()

    predictions_out = generate_predictions(today_games_out, cli_args.model)

    if predictions_out.empty:
        logger.warning("No predictions generated today.")
        pd.DataFrame(
            columns=[
                "GAME_ID",
                "TEAM_NAME",
                "OPPONENT_TEAM_NAME",
                "win_proba",
                "win_pred",
                "prediction_confidence",
            ]
        ).to_csv("data/csv/daily_predictions_empty.csv", index=False)
        sys.exit(0)

    log_system_metrics()
    print_summary(predictions_out)
    log_to_mlflow(predictions_out, model_path=cli_args.model)

    if cli_args.run_shap:
        try:
            from src.interpretability.shap_analysis import run_shap
        except ImportError:

            def run_shap(*args):
                logger.warning("SHAP analysis requested but shap module not available.")

        try:
            shap_dir = "data/shap"
            os.makedirs(shap_dir, exist_ok=True)
            logger.info("Running SHAP analysis...")
            run_shap(cli_args.model, cache_file=DATA_FILE, out_dir=shap_dir)
            log_system_metrics()
        except Exception as e:
            logger.warning("SHAP analysis failed: %s", e)

    logger.info("MLflow NBA daily runner finished.")
