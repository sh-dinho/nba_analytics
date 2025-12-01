# core/db_module.py (Updated)
import sqlite3
from config import DB_PATH # Read from config.py

def connect():
    # Use the path defined in config.py
    return sqlite3.connect(DB_PATH) 


def init_db():
    with connect() as con:
        cur = con.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS nba_games (
                game_id TEXT PRIMARY KEY,
                date TEXT,
                season INTEGER,
                home_team TEXT,
                away_team TEXT,
                home_score INTEGER,
                away_score INTEGER,
                winner TEXT
            )
        """)
        con.commit()