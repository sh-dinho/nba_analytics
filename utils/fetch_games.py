import logging
import sqlite3
import pandas as pd
from utils.nba_api import fetch_nba_games
import yaml
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
CONFIG = yaml.safe_load(open(CONFIG_PATH))
DB_PATH = CONFIG["database"]["path"]

def store_games(df: pd.DataFrame):
    """Store NBA games into SQLite."""
    with sqlite3.connect(DB_PATH) as con:
        df.to_sql("nba_games", con, if_exists="replace", index=False)
    logging.info(f"✔ Stored {len(df)} games into DB.")

if __name__ == "__main__":
    season = 2025
    logging.info(f"🚀 Fetching NBA games for {season} season")
    df_games = fetch_nba_games(season)
    store_games(df_games)
