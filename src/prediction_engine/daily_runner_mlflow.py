# ============================================================
# File: src/prediction_engine/daily_runner_mflow.py
# Purpose: Run daily NBA predictions and log to MLflow
# Project: nba_analysis
# Version: 1.0
# ============================================================

import mlflow
import pandas as pd

from src.prediction_engine.game_features import generate_features_for_games
from src.prediction_engine.predictor import NBAPredictor


def daily_runner_mlflow(model_path, game_date):
    """
    Generate daily predictions and log to MLflow
    game_date: string YYYY-MM-DD
    """
    # Dummy example: Replace with API fetch if needed
    game_data_list = [
        {
            "GAME_ID": "001",
            "TEAM_ID": 1610612737,
            "POINT_SPREAD": -5,
            "OVER_UNDER": 220,
            "TOP_SCORER_20PTS": 1,
        }
    ]
    features = generate_features_for_games(game_data_list)

    predictor = NBAPredictor(model_path)
    features["proba"] = predictor.predict_proba(features)
    features["label"] = predictor.predict_label(features)

    with mlflow.start_run(run_name=f"daily_pred_{game_date}"):
        mlflow.log_artifact(model_path, artifact_path="model")
        csv_path = f"predictions_{game_date}.csv"
        features.to_csv(csv_path, index=False)
        mlflow.log_artifact(csv_path, artifact_path="predictions")

    return features
