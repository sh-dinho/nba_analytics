import sqlite3
import logging
from pathlib import Path
from utils.nba_api import fetch_nba_games
import yaml

# Logging config
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Load config
CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

DB_PATH = CONFIG["database"]["path"]

# Ensure database exists
Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)

def create_games_table():
    """Create nba_games table if not exists"""
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS nba_games (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            home_team TEXT,
            visitor_team TEXT,
            home_score INTEGER,
            visitor_score INTEGER,
            winner TEXT
        )
    """)
    con.commit()
    con.close()

def store_games(df):
    """Store games DataFrame into sqlite db"""
    if df.empty:
        logging.error("❌ No NBA game data to store.")
        return

    con = sqlite3.connect(DB_PATH)
    df.to_sql("nba_games", con, if_exists="replace", index=False)
    con.close()
    logging.info(f"✔ Stored {len(df)} games in database")

def main():
    create_games_table()
    df_games = fetch_nba_games(season=2025)
    if df_games.empty:
        logging.error("❌ No NBA game data found. Cannot proceed.")
        return
    store_games(df_games)

if __name__ == "__main__":
    main()
