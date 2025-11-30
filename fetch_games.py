import sqlite3
import logging
from utils.nba_api import fetch_nba_games
import yaml
import os

logging.basicConfig(level=logging.INFO)

CONFIG = yaml.safe_load(open("config.yaml"))
DB_PATH = CONFIG["database"]["path"]

def create_tables():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS nba_games (
            id INTEGER PRIMARY KEY,
            date TEXT,
            home_team TEXT,
            home_score INTEGER,
            visitor_team TEXT,
            visitor_score INTEGER
        )
    """)
    con.commit()
    con.close()

def store_games(df):
    con = sqlite3.connect(DB_PATH)
    df_db = df[['id', 'date', 'home_team', 'home_score', 'visitor_team', 'visitor_score']]
    df_db.to_sql("nba_games", con, if_exists="replace", index=False)
    con.close()
    logging.info(f"✅ Stored {len(df_db)} NBA games")

if __name__ == "__main__":
    create_tables()
    season = 2025
    logging.info(f"🚀 Fetching NBA games for {season} season")
    df_games = fetch_nba_games(season)
    store_games(df_games)
