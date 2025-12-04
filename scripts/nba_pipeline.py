# ============================================================
# File: scripts/nba_pipeline.py
# Purpose: Feature engineering for XGBoost models
# ============================================================

import pandas as pd
import numpy as np
from datetime import timedelta
from core.config import EV_THRESHOLD

import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder
import requests
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Ensure your API key for Odds-API is loaded from the environment
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
if not ODDS_API_KEY:
    raise ValueError("ODDS_API_KEY is missing from the .env file")

# =====================
# Load Season Data
# =====================

def load_season_data(season_label: str):
    """
    Function to load NBA game data, player stats, and odds data for a given season.
    
    Args:
    season_label (str): The season label in the format "YYYY-YY", e.g., "2023-24".
    
    Returns:
    games_df (DataFrame): DataFrame containing NBA games data.
    stats_df (DataFrame): DataFrame containing NBA player stats.
    odds_data (DataFrame): DataFrame containing NBA odds data.
    """
    
    # STEP 1: Load Game Data (from nba_api)
    game_finder = leaguegamefinder.LeagueGameFinder(season_nullable=season_label)
    games_df = game_finder.get_data_frames()[0]

    # STEP 2: Load Player Stats (from nba_api)
    # You can customize this part to fetch player stats using other endpoints if needed.
    player_stats_df = pd.DataFrame()  # Placeholder for player stats data (You can fetch player data here)
    
    # STEP 3: Load Odds Data (from Odds-API)
    odds_url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
    odds_params = {
        "apiKey": ODDS_API_KEY,
        "date": season_label,  # Use the season_label as the date filter if necessary
        "regions": "us",  # Adjust for specific regions as required
    }
    
    # Get odds data for the season
    response = requests.get(odds_url, params=odds_params)
    
    if response.status_code == 200:
        odds_data = response.json()
        odds_df = pd.DataFrame(odds_data)
    else:
        print(f"Error fetching odds data: {response.status_code}")
        odds_df = pd.DataFrame()

    return games_df, player_stats_df, odds_df


# =====================
# Example Usage: Fetch Data for the Season
# =====================

# Replace with your desired season
season_label = "2025-26"  # Example season

try:
    # Fetch data
    games_df, stats_df, odds_data = load_season_data(season_label)
    
    # Print first few rows of data (for debugging purposes)
    print("Games Data:")
    print(games_df.head())

    print("\nPlayer Stats Data:")
    print(stats_df.head())

    print("\nOdds Data:")
    print(odds_data.head())
except Exception as e:
    print(f"An error occurred: {e}")

# =====================
# Additional Pipeline Code for Processing (if needed)
# =====================

# Here, you can add additional processing steps if needed, like merging the data, 
# creating features, or passing the data to a model for predictions.
