import logging
from pathlib import Path
import sqlite3
import pandas as pd
import requests
from utils.nba_api import fetch_nba_games

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Paths
BASE_DIR = Path(__file__).parent.parent.resolve()  # project root
CONFIG_PATH = BASE_DIR / "config.yaml"
DB_PATH = BASE_DIR / "nba_analytics.db"

# Load config
import yaml
if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Config file not found at {CONFIG_PATH}")
CONFIG = yaml.safe_load(open(CONFIG_PATH))

def store_games(df: pd.DataFrame):
    if df.empty:
        logging.error("❌ No NBA game data found. Cannot proceed.")
        return
    with sqlite3.connect(DB_PATH) as con:
        df.to_sql("nba_games", con, if_exists="replace", index=False)
    logging.info("✔ NBA games stored successfully.")

if __name__ == "__main__":
    season = 2025
    logging.info(f"🚀 Fetching NBA games for {season} season")
    try:
        df_games = fetch_nba_games(season)
        store_games(df_games)
    except Exception as e:
        logging.error(f"Failed to fetch/store games: {e}")
