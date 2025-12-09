# ============================================================
# Path: src/prediction_engine/game_features.py
# Filename: game_features.py
# Author: Your Team
# Date: December 2025
# Purpose: Functions to fetch NBA game IDs and transform
#          raw game stats into model-ready features.
# ============================================================

import pathlib
import pandas as pd
from nba_api.stats.endpoints import boxscoretraditionalv2, leaguegamefinder

# -----------------------------
# Local cache path (used in tests)
# -----------------------------
_cache_path = pathlib.Path("results/cache")
_cache_path.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Fetch game IDs for a season
# -----------------------------
def fetch_season_games(season_year: int, limit: int = 10):
    """
    Fetch a list of NBA game IDs for a given season.

    Args:
        season_year (int): The starting year of the season (e.g., 2023 for 2023-24).
        limit (int): Number of game IDs to return.

    Returns:
        list[str]: A list of game IDs.
    """
    season_str = f"{season_year}-{str(season_year+1)[-2:]}"
    gamefinder = leaguegamefinder.LeagueGameFinder(season_nullable=season_str)
    games = gamefinder.get_data_frames()[0]
    game_ids = games["GAME_ID"].unique().tolist()
    return game_ids[:limit]

# -----------------------------
# Fetch features for a single game
# -----------------------------
def fetch_game_features(game_id: str) -> pd.DataFrame:
    """
    Fetch box score stats for a given game and return model-ready features.

    Args:
        game_id (str): NBA game ID.

    Returns:
        pd.DataFrame: DataFrame with selected features and win/loss label.
    """
    boxscore = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
    stats = boxscore.get_data_frames()[0]

    # Define the columns we want to aggregate
    required_columns = ["PTS", "REB", "AST", "FG_PCT", "FT_PCT", "PLUS_MINUS", "TOV"]

    # Build aggregation dict only for columns that exist
    agg_dict = {}
    for col in required_columns:
        if col in stats.columns:
            if col in ["FG_PCT", "FT_PCT"]:
                agg_dict[col] = "mean"
            else:
                agg_dict[col] = "sum"

    team_stats = stats.groupby("TEAM_ID").agg(agg_dict).reset_index()

    # Ensure all required columns exist, fill with default if missing
    for col in required_columns:
        if col not in team_stats.columns:
            team_stats[col] = 0

    # Add win/loss label: positive PLUS_MINUS → win
    team_stats["win"] = (team_stats["PLUS_MINUS"] > 0).astype(int)

    return team_stats

# -----------------------------
# Generate features for multiple games
# -----------------------------
def generate_features_for_games(game_ids: list[str]) -> pd.DataFrame:
    """
    Generate features for a list of game IDs.

    Args:
        game_ids (list[str]): List of NBA game IDs.

    Returns:
        pd.DataFrame: Concatenated features for all games.
    """
    features = pd.concat([fetch_game_features(gid) for gid in game_ids], ignore_index=True)
    return features
