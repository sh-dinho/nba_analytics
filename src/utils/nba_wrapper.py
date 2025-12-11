# ============================================================
# File: src/utils/nba_api_wrapper.py
# Purpose: Wrapper for nba_api endpoints to fetch games & stats
# Project: nba_analysis
# Version: 1.1 (adds dependencies section + defensive handling)
#
# Dependencies:
# - logging (standard library)
# - datetime (standard library)
# - pandas
# - nba_api.stats.endpoints.leaguegamefinder
# - nba_api.stats.endpoints.scoreboard
# ============================================================

import logging
from datetime import datetime

import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder, scoreboard

# Set up logging for this module
logger = logging.getLogger("nba_api_wrapper")


def fetch_season_games(season: str) -> pd.DataFrame:
    """
    Fetch all games for a given season using nba_api.

    Args:
        season (str): The NBA season (e.g., '2022-23').

    Returns:
        pd.DataFrame: A DataFrame containing game data including GAME_DATE, TEAM_NAME,
                      MATCHUP, GAME_ID, TEAM_ID, OPPONENT_TEAM_ID, POINTS, and TARGET.
    """
    logger.info(f"Fetching games for season {season}...")
    try:
        gamefinder = leaguegamefinder.LeagueGameFinder(season_nullable=season)
        df = gamefinder.get_data_frames()[0]

        # Keep relevant columns and rename for clarity
        df = df[
            [
                "GAME_DATE",
                "TEAM_NAME",
                "MATCHUP",
                "GAME_ID",
                "TEAM_ID",
                "OPPONENT_TEAM_ID",
                "PTS",
                "WL",
            ]
        ]
        df.rename(columns={"PTS": "POINTS", "WL": "TARGET"}, inplace=True)
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")  # Safe conversion
        logger.info(f"Fetched {len(df)} games for season {season}.")
        return df
    except Exception as e:
        logger.error(f"Error fetching season {season}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame if there's an error


def fetch_today_games() -> pd.DataFrame:
    """
    Fetch today's NBA games using the scoreboard endpoint.

    Returns:
        pd.DataFrame: A DataFrame containing today's games data ready for feature generation.
    """
    today_str = datetime.now().strftime("%Y-%m-%d")
    logger.info(f"Fetching today's games for {today_str}...")
    try:
        sb = scoreboard.Scoreboard(game_date=today_str)
        games = sb.get_data_frames()[0]  # Get the first dataframe from the API response
    except Exception as e:
        logger.error(f"Error fetching today's scoreboard: {e}")
        return pd.DataFrame()  # Return an empty DataFrame if there's an error

    if games.empty:
        logger.info("No NBA games today.")
        return pd.DataFrame()  # Return an empty DataFrame if there are no games

    # Normalize column names for feature generation
    games = games.rename(
        columns={
            "GAME_ID": "GAME_ID",
            "HOME_TEAM_ID": "HOME_TEAM_ID",
            "VISITOR_TEAM_ID": "AWAY_TEAM_ID",
        }
    )

    logger.info(f"Fetched {len(games)} games for {today_str}.")
    return games
