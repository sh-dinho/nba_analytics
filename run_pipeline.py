# ============================================================
# File: daily_runner.py
# Purpose: Daily NBA prediction runner with enhanced features
# Version: 1.0
# ============================================================

import pandas as pd
import numpy as np
import datetime
import logging
import json
from pathlib import Path

from src.prediction_engine.predictor import NBAPredictor
from src.feature_engineering.generate_features import generate_features_for_games
from src.utils.team_player_mapping import TEAM_ID_NAME, PLAYER_ID_NAME
from nba_api.stats.endpoints import leaguegamefinder, boxscoretraditionalv2

# ----------------------------
# Utility functions
# ----------------------------

def season_string(year: int) -> str:
    return f"{year}-{str(year+1)[-2:]}"

def fetch_games_by_date(year: int, game_date: str) -> list[dict]:
    """Return raw game data for a specific date."""
    finder = leaguegamefinder.LeagueGameFinder(season_nullable=season_string(year))
    df = finder.get_data_frames()[0]
    df_today = df[df["GAME_DATE"] == game_date]
    games = df_today.to_dict(orient="records")
    return games

def fetch_player_stats(game_id: str, min_points: int = 20) -> dict:
    """Return a dict of player stats: players with >= min_points."""
    try:
        box = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
        df = box.get_data_frames()[0]
        df_top = df[df["PTS"] >= min_points]
        return {"players_20pts_plus": df_top[["PLAYER_ID", "PLAYER_NAME", "PTS"]].to_dict(orient="records")}
    except Exception as e:
        logging.warning("Failed to fetch player stats for game %s: %s", game_id, e)
        return {"players_20pts_plus": []}

def enrich_with_betting(raw_games: list[dict]) -> list[dict]:
    """Add betting features: point spread, over/under, home/away."""
    for g in raw_games:
        g["POINT_SPREAD"] = g.get("POINT_SPREAD", 0)
        g["OVER_UNDER"] = g.get("OVER_UNDER", 0)
        g["HOME"] = int(g.get("MATCHUP", "").startswith("vs"))
    return raw_games

def map_ids_to_names(df: pd.DataFrame) -> pd.DataFrame:
    """Map TEAM_ID / PLAYER_ID to names for frontend."""
    df["TEAM_NAME"] = df["TEAM_ID"].map(TEAM_ID_NAME).fillna("Unknown Team")
    if "players_20pts_plus" in df.columns:
        def map_players(lst):
            for p in lst:
                p["PLAYER_NAME"] = PLAYER_ID_NAME.get(p["PLAYER_ID"], p.get("PLAYER_NAME", "Unknown Player"))
            return lst
        df["players_20pts_plus"] = df["players_20pts_plus"].apply(map_players)
    return df

# ----------------------------
# Main runner
# ----------------------------

def daily_runner(model_path: str, game_date: str = None, year: int = 2025, threshold: float = 0.5):
    # Step 1: Fetch raw games
    if not game_date:
        game_date = datetime.date.today().isoformat()
    raw_games = fetch_games_by_date(year, game_date)
    if not raw_games:
        logging.warning("No games found for %s", game_date)
        return []

    # Step 2: Add betting + home/away
    raw_games = enrich_with_betting(raw_games)

    # Step 3: Fetch player stats
    for g in raw_games:
        g.update(fetch_player_stats(g.get("GAME_ID", "unknown")))

    # Step 4: Generate features
    features_df = generate_features_for_games(raw_games)

    # Step 5: Load predictor and predict
    predictor = NBAPredictor(model_path=model_path)
    X = features_df.drop(columns=["win"], errors="ignore")
    features_df["win_proba"] = predictor.predict_proba(X)
    features_df["win_label"] = predictor.predict_label(X, threshold=threshold)

    # Step 6: Map IDs to names for frontend
    features_df = map_ids_to_names(features_df)

    # Step 7: Convert to JSON
    results_json = features_df.to_dict(orient="records")
    print(json.dumps(results_json, indent=2))
    return results_json

# ----------------------------
# CLI support
# ----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Daily NBA Predictor Runner")
    parser.add_argument("--model", required=True, help="Path to trained model (.pkl)")
    parser.add_argument("--date", help="Game date YYYY-MM-DD (default today)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for win label")
    args = parser.parse_args()

    daily_runner(model_path=args.model, game_date=args.date, threshold=args.threshold)
