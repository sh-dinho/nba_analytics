import sqlite3
import logging
from utils.nba_api import fetch_nba_games
from datetime import datetime
import yaml

CONFIG = yaml.safe_load(open("config.yaml"))
DB_PATH = CONFIG["database"]["path"]

def store_games(df):
    conn = sqlite3.connect(DB_PATH)
    df["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df.to_sql("nba_games", conn, if_exists="replace", index=False)
    conn.close()
    logging.info(f"Stored {len(df)} games in DB")

def main():
    logging.info("🚀 Fetching NBA games...")
    df = fetch_nba_games()
    if df.empty:
        logging.error("❌ No NBA game data found. Cannot proceed.")
        return
    store_games(df)

if __name__ == "__main__":
    main()
