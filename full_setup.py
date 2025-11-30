import sqlite3
import logging
import os

logging.basicConfig(level=logging.INFO)

DB_PATH = "nba_analytics.db"

def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS nba_games (
        game_id TEXT PRIMARY KEY,
        home_team TEXT,
        away_team TEXT,
        home_team_score INTEGER,
        away_team_score INTEGER,
        home_team_rest INTEGER,
        away_team_rest INTEGER,
        home_win INTEGER
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS retrain_history (
        Timestamp TEXT,
        ModelType TEXT,
        Status TEXT
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS daily_picks (
        home_team TEXT,
        away_team TEXT,
        ev REAL
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS bankroll_tracker (
        Timestamp TEXT,
        CurrentBankroll REAL,
        ROI REAL
    )
    """)
    con.commit()
    con.close()
    logging.info("✔ Database initialized")

if __name__ == "__main__":
    init_db()
