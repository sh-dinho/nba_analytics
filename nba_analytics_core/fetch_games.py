from nba_api.stats.endpoints import leaguegamefinder
import pandas as pd

def fetch_historical_games(season="2024-25"):
    gamefinder = leaguegamefinder.LeagueGameFinder(season_nullable=season)
    games = gamefinder.get_data_frames()[0]
    return games