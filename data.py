import sqlite3
import pandas as pd
import yaml
import logging

logging.basicConfig(level=logging.INFO)

with open("config.yaml") as f:
    CONFIG = yaml.safe_load(f)
DB_PATH = CONFIG["database"]["path"]

_cache = {}

def connect():
    return sqlite3.connect(DB_PATH)

def fetch_historical_games(season: int) -> pd.DataFrame:
    """Fetch games for a given season, using cache if available."""
    if season in _cache:
        return _cache[season]

    with connect() as con:
        df = pd.read_sql(
            "SELECT * FROM nba_games WHERE season=? ORDER BY date ASC",
            con,
            params=(season,)
        )
    _cache[season] = df
    logging.info(f"Loaded {len(df)} games for season {season}")
    return df

def preload_seasons(seasons: list[int]):
    """Preload multiple seasons into cache at startup."""
    for season in seasons:
        if season not in _cache:
            with connect() as con:
                df = pd.read_sql(
                    "SELECT * FROM nba_games WHERE season=? ORDER BY date ASC",
                    con,
                    params=(season,)
                )
            _cache[season] = df
            logging.info(f"Preloaded {len(df)} games for season {season}")

def get_cached_season(season: int) -> pd.DataFrame:
    """Return cached season data if available."""
    return _cache.get(season, pd.DataFrame())