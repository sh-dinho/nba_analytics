import os
import sqlite3
import pandas as pd
import requests
import logging
from datetime import datetime
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Config
ROOT = os.path.dirname(os.path.dirname(__file__))
CONFIG_PATH = os.path.join(ROOT, "config.yaml")
CONFIG = yaml.safe_load(open(CONFIG_PATH))

DB_PATH = CONFIG["database"]["path"]
API_URL = CONFIG["nba_api"]["base_url"]
SEASON = CONFIG["nba_api"]["season"]

# ============================================================
# DATABASE FUNCTIONS
# ============================================================
def init_db():
    """Create nba_games table if it does not exist"""
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
        CREATE TABLE IF NOT EXISTS nba_games (
            game_id TEXT PRIMARY KEY,
            date TEXT,
            home_team TEXT,
            visitor_team TEXT,
            home_score INTEGER,
            visitor_score INTEGER,
            home_win INTEGER,
            created_at TEXT
        )
        """)
    logging.info("✅ Database initialized or already exists.")


def store_games(df: pd.DataFrame):
    """Store new games into DB (deduplicate by game_id)"""
    if df.empty:
        logging.warning("❌ No data to store")
        return

    df["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with sqlite3.connect(DB_PATH) as con:
        for _, row in df.iterrows():
            try:
                con.execute("""
                    INSERT OR IGNORE INTO nba_games
                    (game_id, date, home_team, visitor_team, home_score, visitor_score, home_win, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    row["game_id"], row["date"], row["home_team"], row["visitor_team"],
                    row["home_score"], row["visitor_score"], row["home_win"], row["created_at"]
                ))
            except Exception as e:
                logging.error(f"Failed to insert row {row['game_id']}: {e}")
    logging.info(f"✅ Stored {len(df)} games to database.")


# ============================================================
# NBA API FETCH
# ============================================================
def fetch_nba_games():
    """
    Fetch games for the season from API.
    Replace with a real API or mock endpoint.
    """
    logging.info(f"🚀 Fetching NBA games for {SEASON} season from API")
    url = f"{API_URL}/games?season={SEASON}"

    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
        if not data.get("games"):
            logging.error("❌ API returned no games")
            return pd.DataFrame()
    except Exception as e:
        logging.error(f"Failed to fetch games: {e}")
        return pd.DataFrame()

    # Transform into DataFrame
    games = []
    for g in data["games"]:
        games.append({
            "game_id": g["id"],
            "date": g["date"],
            "home_team": g["home_team"],
            "visitor_team": g["visitor_team"],
            "home_score": g.get("home_score", None),
            "visitor_score": g.get("visitor_score", None),
            "home_win": int(g.get("home_score", 0) > g.get("visitor_score", 0)) if g.get("home_score") else None
        })

    df = pd.DataFrame(games)
    logging.info(f"✅ Fetched {len(df)} games")
    return df


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    init_db()
    df_games = fetch_nba_games()
    if df_games.empty:
        logging.error("❌ No NBA game data found. Cannot proceed.")
    else:
        store_games(df_games)
