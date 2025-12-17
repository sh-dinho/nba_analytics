# ============================================================
# File: src/api/nba_api_client.py
# Purpose: Fetch live NBA games for today from the NBA API
# Author: Your Name or Team Name
# Date: 2025-12-17
# ============================================================

import logging
from datetime import datetime
import pandas as pd
import requests
import time
from requests.exceptions import RequestException

# Set up logging for the module
logger = logging.getLogger("nba_api_client")

# NBA API URL for retrieving game data
NBA_API_URL = (
    "https://data.nba.com/data/10s/v2015/json/mobile_teams/nba/latest/games.json"
)

# Retry configuration for handling API failures
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds


def fetch_today_games() -> pd.DataFrame:
    """
    Fetch today's NBA games from the official NBA API with retries.

    This function makes an API call to the NBA's official API to retrieve
    details about all games scheduled for today. It retries the request
    up to a maximum number of times if it encounters a failure (e.g., timeout,
    connection issues). The results are returned in a DataFrame, containing
    information about the game IDs, home and away teams, start times, and
    scores for today's games.

    Returns:
        pd.DataFrame: A DataFrame with columns:
            ['GAME_ID', 'HOME_TEAM', 'AWAY_TEAM', 'START_TIME', 'STATUS',
             'HOME_TEAM_SCORE', 'AWAY_TEAM_SCORE']
    """
    for attempt in range(MAX_RETRIES):
        try:
            # Send the request to the NBA API
            response = requests.get(NBA_API_URL, timeout=10)
            response.raise_for_status()  # Raise an error for bad status codes
            data = response.json()

            # Navigate to today's games
            games_today = []
            today_str = datetime.now().strftime("%Y-%m-%d")

            # Iterate over the games and select only today's games
            for game in data.get("g", []):
                game_date = game.get("gdte")  # date in YYYY-MM-DD
                if game_date == today_str:
                    games_today.append(
                        {
                            "GAME_ID": game.get("gid"),
                            "HOME_TEAM": game.get("h", {}).get("ta"),
                            "AWAY_TEAM": game.get("v", {}).get("ta"),
                            "START_TIME": game.get("stt"),
                            "STATUS": game.get("st"),
                            "HOME_TEAM_SCORE": game.get("h", {}).get("s", 0),
                            "AWAY_TEAM_SCORE": game.get("v", {}).get("s", 0),
                        }
                    )

            # Convert the list of games to a DataFrame
            df = pd.DataFrame(games_today)

            # Check if no games were found
            if df.empty:
                logger.warning("No NBA games found for today.")
            return df

        except RequestException as e:
            # Log the error and retry the request if necessary
            logger.error(f"Attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)  # Wait before retrying
            else:
                logger.error("No NBA games found after multiple attempts.")
                return pd.DataFrame()  # Return empty DataFrame after max retries
