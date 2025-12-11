# ============================================================
# File: src/prediction_engine/daily_runner.py
# Purpose: Canonical daily runner for NBA predictions
# Project: nba_analysis
# Version: 2.0 (returns features, predictions, player info)
# ============================================================

import logging
import pandas as pd

from src.api.nba_api_client import fetch_season_games
from src.features.feature_engineering import generate_features_for_games
from src.prediction_engine.predictor import Predictor

logger = logging.getLogger("prediction_engine.daily_runner")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def run_daily_predictions(model_path, season=2025, limit=10):
    """Fetch games, generate features, run predictions, return DataFrames."""
    games = fetch_season_games(season, limit=limit)
    if games is None or games.empty:
        logger.warning("No games found for season %s", season)
        return None, None, None

    features = generate_features_for_games(games.to_dict(orient="records"))
    if features is None or features.empty:
        logger.warning("No features generated.")
        return None, None, None

    predictor = Predictor(model_path=model_path)
    X = features.drop(columns=["win"], errors="ignore")

    predictions_df = pd.DataFrame(
        {
            "GAME_ID": features["GAME_ID"],
            "win_proba": predictor.predict_proba(X),
            "win_pred": predictor.predict_label(X),
        }
    )

    # Placeholder: player info could come from another API or join
    player_info_df = pd.DataFrame(columns=["GAME_ID", "TEAM_ID", "PlayerNames"])

    return features, predictions_df, player_info_df
