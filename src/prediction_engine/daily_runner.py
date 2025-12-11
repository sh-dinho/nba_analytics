# ============================================================
# File: src/prediction_engine/daily_runner.py
# Purpose: Core logic for daily predictions
# ============================================================

import logging
import pandas as pd
from nba_api.stats.endpoints import commonteamroster
from src.api.nba_api_client import fetch_today_games
from src.model_training.utils import load_model, build_features

logger = logging.getLogger("prediction_engine.daily_runner")


def fetch_team_roster(team_id: int, season: int) -> pd.DataFrame:
    """
    Fetch roster info for a given team and season.
    Returns DataFrame with PLAYER_ID, PLAYER, POSITION, HEIGHT, WEIGHT, BIRTH_DATE, EXP.
    """
    try:
        roster = commonteamroster.CommonTeamRoster(team_id=team_id, season=season)
        df = roster.get_data_frames()[0]
        return df[
            ["PLAYER_ID", "PLAYER", "POSITION", "HEIGHT", "WEIGHT", "BIRTH_DATE", "EXP"]
        ]
    except Exception as e:
        logger.warning("Failed to fetch roster for team %s: %s", team_id, e)
        return pd.DataFrame()


def run_daily_predictions(model: str, season: int, limit: int = 10):
    """
    Run daily predictions.
    Always returns 3 DataFrames: (features_df, predictions_df, player_info_df).
    If no games are found, returns empty DataFrames.
    """
    try:
        # --- Fetch today's games (or next available) ---
        games_df = fetch_today_games()
        if games_df.empty:
            logger.warning("No games found today. Returning empty DataFrames.")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        # --- Build features ---
        features_df = build_features(games_df, season=season)

        # --- Load model ---
        model_obj = load_model(model)

        # --- Generate predictions ---
        y_pred = model_obj.predict(features_df)
        y_prob = None
        try:
            y_prob = model_obj.predict_proba(features_df)[:, 1]
        except Exception:
            pass

        predictions_df = games_df.copy()
        predictions_df["PREDICTED_WIN"] = y_pred
        if y_prob is not None:
            predictions_df["WIN_PROBABILITY"] = y_prob

        # --- Player info (rosters for all teams in games) ---
        player_info_frames = []
        if "HOME_TEAM_ID" in games_df.columns and "VISITOR_TEAM_ID" in games_df.columns:
            team_ids = pd.concat(
                [games_df["HOME_TEAM_ID"], games_df["VISITOR_TEAM_ID"]]
            ).unique()
            for team_id in team_ids:
                roster_df = fetch_team_roster(int(team_id), season)
                if not roster_df.empty:
                    roster_df["TEAM_ID"] = team_id
                    player_info_frames.append(roster_df)

        player_info_df = (
            pd.concat(player_info_frames, ignore_index=True)
            if player_info_frames
            else pd.DataFrame()
        )

        # --- Limit results ---
        if limit:
            predictions_df = predictions_df.head(limit)

        return features_df, predictions_df, player_info_df

    except Exception as e:
        logger.error("Daily predictions failed: %s", e)
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
