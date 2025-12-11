# ============================================================
# File: src/utils/nba_api_wrapper.py
# Purpose: Wrapper for nba_api endpoints to fetch games & stats
# Project: nba_analysis
# Version: 1.3 (adds combined helper fetch_games)
# ============================================================

import logging
from datetime import datetime
import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder, scoreboard

logger = logging.getLogger("nba_api_wrapper")

EXPECTED_COLS = [
    "GAME_DATE",
    "TEAM_NAME",
    "MATCHUP",
    "GAME_ID",
    "TEAM_ID",
    "OPPONENT_TEAM_ID",
    "POINTS",
    "TARGET",
]
EXPECTED_TODAY_COLS = ["GAME_ID", "HOME_TEAM_ID", "AWAY_TEAM_ID"]


def fetch_season_games(season: str) -> pd.DataFrame:
    """Fetch all games for a given season using nba_api."""
    if not isinstance(season, str):
        raise TypeError("season must be a string like '2022-23'")

    logger.info("Fetching games for season %s...", season)
    try:
        gamefinder = leaguegamefinder.LeagueGameFinder(season_nullable=season)
        df = gamefinder.get_data_frames()[0]

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
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
        df["TARGET"] = df["TARGET"].map({"W": 1, "L": 0})

        logger.info("Fetched %d games for season %s.", len(df), season)
        return df
    except Exception as e:
        logger.error("Error fetching season %s: %s", season, e)
        return pd.DataFrame(columns=EXPECTED_COLS)


def fetch_today_games() -> pd.DataFrame:
    """Fetch today's NBA games using the scoreboard endpoint."""
    today_str = datetime.now().strftime("%Y-%m-%d")
    logger.info("Fetching today's games for %s...", today_str)
    try:
        sb = scoreboard.Scoreboard(game_date=today_str)
        games = sb.get_data_frames()[0]
    except Exception as e:
        logger.error("Error fetching today's scoreboard: %s", e)
        return pd.DataFrame(columns=EXPECTED_TODAY_COLS)

    if games.empty:
        logger.info("No NBA games today.")
        return pd.DataFrame(columns=EXPECTED_TODAY_COLS)

    games = games.rename(columns={"VISITOR_TEAM_ID": "AWAY_TEAM_ID"})
    games = games[["GAME_ID", "HOME_TEAM_ID", "AWAY_TEAM_ID"]]

    logger.info("Fetched %d games for %s.", len(games), today_str)
    return games


def fetch_games(season: str = None) -> pd.DataFrame:
    """
    Combined helper: fetch either today's games (if season is None) or all games for a season.

    Args:
        season (str, optional): NBA season string (e.g., '2022-23'). If None, fetch today's games.

    Returns:
        pd.DataFrame: Games data with consistent schema.
    """
    if season is None:
        return fetch_today_games()
    return fetch_season_games(season)
