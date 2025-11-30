import sqlite3
import pandas as pd
import yaml
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)

# --- Load config safely ---
try:
    with open("config.yaml") as f:
        CONFIG = yaml.safe_load(f)
except FileNotFoundError:
    logging.warning("⚠️ config.yaml not found, using defaults.")
    CONFIG = {"database": {"path": "bets.db"}}

DB_PATH = CONFIG["database"]["path"]

def connect():
    """Connect to SQLite database with row factory."""
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con

def get_todays_games() -> pd.DataFrame:
    """
    Fetch today's NBA games from the database.
    Returns DataFrame with columns: home_team, away_team, date, home_score, away_score, winner.
    """
    today = datetime.now().strftime("%Y-%m-%d")
    query = "SELECT * FROM nba_games WHERE date=? ORDER BY home_team ASC"

    with connect() as con:
        df = pd.read_sql(query, con, params=(today,))

    if df.empty:
        logging.info(f"No games found for today ({today})")
    return df

def store_games(games: pd.DataFrame):
    """
    Store new games into the nba_games table.
    Expects DataFrame with columns: game_id, date, season, home_team, away_team, home_score, away_score, winner.
    """
    if games is None or games.empty:
        logging.warning("No games to store.")
        return

    with connect() as con:
        cur = con.cursor()
        for _, row in games.iterrows():
            cur.execute("""
                INSERT OR REPLACE INTO nba_games
                (game_id, date, season, home_team, away_team, home_score, away_score, winner)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row["game_id"],
                row["date"],
                row["season"],
                row["home_team"],
                row["away_team"],
                row["home_score"],
                row["away_score"],
                row["winner"]
            ))
        con.commit()
    logging.info(f"✔ Stored {len(games)} games into DB")