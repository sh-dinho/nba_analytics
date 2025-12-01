# nba_analytics_core/data.py
import logging
import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder

def fetch_historical_games(season: str = "2024-25") -> pd.DataFrame:
    logging.info(f"Fetching NBA games for season {season}...")
    gamefinder = leaguegamefinder.LeagueGameFinder(season_nullable=season)
    games = gamefinder.get_data_frames()[0]
    logging.info(f"✔ Retrieved {len(games)} games from NBA API")
    return games