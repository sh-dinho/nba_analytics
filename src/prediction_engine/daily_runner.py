# ============================================================
# File: src/prediction_engine/daily_runner.py
# Purpose: Run daily NBA predictions using schedule-based features
# Project: nba_analysis
# Version: 2.1 (aligned with schedule features)
# ============================================================

import logging
import pandas as pd

from src.api.nba_api_client import fetch_today_games
from src.features.feature_engineering import generate_features_for_games
from src.prediction_engine.predictor import Predictor

logger = logging.getLogger("prediction_engine.daily_runner")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def run_daily_predictions(model_path, season=2025, limit=10):
    """
    Fetch today's games, generate schedule-based features, run predictions.
    Returns (features_df, predictions_df).
    """
    games = fetch_today_games()
    if games is None or games.empty:
        logger.warning("No games found for today.")
        return None, None

    if limit:
        games = games.head(limit)

    features = generate_features_for_games(games)
    if features is None or features.empty:
        logger.warning("No features generated.")
        return None, None

    predictor = Predictor(model_path=model_path)

    # Drop target if present
    X = features.drop(columns=["win"], errors="ignore")

    predictions_df = pd.DataFrame(
        {
            "GAME_DATE": features["GAME_DATE"],
            "TEAM_NAME": features["TEAM_NAME"],
            "HOME_GAME": features.get("HOME_GAME", None),
            "win_proba": predictor.predict_proba(X),
            "win_pred": predictor.predict_label(X),
        }
    )

    logger.info("Generated predictions for %d games", len(predictions_df))
    return features, predictions_df
