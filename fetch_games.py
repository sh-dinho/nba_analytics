import sqlite3
from utils.nba_api import fetch_nba_games
import logging

DB_PATH = "nba_analytics.db"
SEASON = 2025

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def store_games(df):
    if df.empty:
        logging.error("❌ No NBA game data found. Cannot proceed.")
        return
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS nba_games (
            id INTEGER PRIMARY KEY,
            date TEXT,
            HOME_TEAM TEXT,
            VISITOR_TEAM TEXT,
            HOME_SCORE INTEGER,
            VISITOR_SCORE INTEGER
        )
    """)
    df.to_sql("nba_games", con, if_exists="replace", index=False)
    con.close()
    logging.info(f"✔ {len(df)} games stored in database")

if __name__ == "__main__":
    df_games = fetch_nba_games(SEASON)
    store_games(df_games)
