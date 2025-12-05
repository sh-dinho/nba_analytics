# ============================================================
# File: scripts/nba_pipeline.py
# Purpose: Feature engineering for XGBoost models
# ============================================================

import os
import requests
import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
if not ODDS_API_KEY:
    raise ValueError("ODDS_API_KEY is missing from the .env file")

# =====================
# Load Season Data
# =====================

def load_season_data(season_label: str):
    """
    Load NBA game data, player stats, and odds data for a given season.

    Args:
        season_label (str): Season label in format "YYYY-YY", e.g., "2023-24".

    Returns:
        games_df (DataFrame): NBA games data
        player_stats_df (DataFrame): NBA player stats
        odds_df (DataFrame): NBA odds data
    """

    # STEP 1: Load Game Data
    game_finder = leaguegamefinder.LeagueGameFinder(season_nullable=season_label)
    games_df = game_finder.get_data_frames()[0]

    # STEP 2: Load Player Stats (placeholder for now)
    player_stats_df = pd.DataFrame()

    # STEP 3: Load Odds Data
    odds_url = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
    odds_params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": "h2h,spreads,totals"
    }

    try:
        response = requests.get(odds_url, params=odds_params)
        response.raise_for_status()
        odds_data = response.json()
        odds_df = pd.DataFrame(odds_data)
    except requests.RequestException as e:
        print(f"Error fetching odds data: {e}")
        odds_df = pd.DataFrame()

    return games_df, player_stats_df, odds_df

# =====================
# Example Usage
# =====================

if __name__ == "__main__":
    season_label = "2025-26"
    try:
        games_df, stats_df, odds_df = load_season_data(season_label)

        print("Games Data:")
        print(games_df.head())

        print("\nPlayer Stats Data:")
        print(stats_df.head())

        print("\nOdds Data:")
        print(odds_df.head())
    except Exception as e:
        print(f"Pipeline error: {e}")