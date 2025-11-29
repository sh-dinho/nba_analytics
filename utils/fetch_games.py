import sqlite3
import logging
from utils.nba_api import fetch_nba_games
import yaml
import os

logging.basicConfig(level=logging.INFO)

CONFIG = yaml.safe_load(open("config.yaml"))
DB_PATH = CONFIG["database"]["path"]

def store_games(df):
    if df.empty:
        logging.error("No NBA game data found. Cannot proceed.")
        return
    with sqlite3.connect(DB_PATH) as con:
        df.to_sql("nba_games", con, if_exists="replace", index=False)
        logging.info("NBA games stored in DB")

if __name__ == "__main__":
    logging.info("Fetching NBA games...")
    df = fetch_nba_games(2025)
    store_games(df)
