# ============================================================
# File: src/utils/nba_api_wrapper.py
# Purpose: Wrapper for nba_api endpoints to fetch games & stats
# ============================================================

import pandas as pd
from datetime import datetime
from nba_api.stats.endpoints import leaguegamefinder, scoreboard
import logging

logger = logging.getLogger("nba_api_wrapper")

def fetch_season_games(season: str) -> pd.DataFrame:
    """
    Fetch all games for a given season using nba_api.
    Returns DataFrame with columns: GAME_DATE, TEAM_NAME, MATCHUP, HOME/AWAY, etc.
    """
    logger.info(f"Fetching games for season {season}...")
    try:
        gamefinder = leaguegamefinder.LeagueGameFinder(season_nullable=season)
        df = gamefinder.get_data_frames()[0]

        # Keep relevant columns
        df = df[['GAME_DATE', 'TEAM_NAME', 'MATCHUP', 'GAME_ID', 'TEAM_ID', 'OPPONENT_TEAM_ID', 'PTS', 'WL']]
        df.rename(columns={"PTS":"POINTS","WL":"TARGET"}, inplace=True)
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        return df
    except Exception as e:
        logger.error(f"Error fetching season {season}: {e}")
        return pd.DataFrame()


def fetch_today_games() -> pd.DataFrame:
    """
    Fetch today's NBA games using the scoreboard endpoint.
    Returns DataFrame ready for feature generation.
    """
    today_str = datetime.now().strftime("%Y-%m-%d")
    try:
        sb = scoreboard.Scoreboard(game_date=today_str)
        games = sb.get_data_frames()[0]
    except Exception as e:
        logger.error(f"Error fetching today's scoreboard: {e}")
        return pd.DataFrame()

    if games.empty:
        logger.info("No NBA games today.")
        return pd.DataFrame()

    # Normalize columns for feature generation
    games.rename(columns={
        "GAME_ID": "GAME_ID",
        "HOME_TEAM_ID": "HOME_TEAM_ID",
        "VISITOR_TEAM_ID": "AWAY_TEAM_ID"
    }, inplace=True)

    return games
