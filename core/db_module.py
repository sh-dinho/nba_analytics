import sqlite3
import yaml
import logging
import os
import pandas as pd
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
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con

def init_schema_version(cur):
    """Ensure schema_version table exists and has an entry."""
    cur.execute("""
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER NOT NULL,
            applied_on TEXT NOT NULL
        )
    """)
    cur.execute("SELECT COUNT(*) FROM schema_version")
    if cur.fetchone()[0] == 0:
        cur.execute("INSERT INTO schema_version (version, applied_on) VALUES (?, datetime('now'))", (1,))

def init_db():
    """Initialize all tables and indexes."""
    with connect() as con:
        cur = con.cursor()
        cur.execute("""CREATE TABLE IF NOT EXISTS nba_games (
            game_id TEXT PRIMARY KEY,
            date TEXT,
            season INTEGER,
            home_team TEXT,
            away_team TEXT,
            home_score INTEGER,
            away_score INTEGER,
            winner TEXT
        )""")
        cur.execute("""CREATE TABLE IF NOT EXISTS daily_picks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Timestamp TEXT,
            home_team TEXT,
            away_team TEXT,
            ev REAL
        )""")
        cur.execute("""CREATE TABLE IF NOT EXISTS bankroll_tracker (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Timestamp TEXT,
            CurrentBankroll REAL,
            ROI REAL
        )""")
        cur.execute("""CREATE TABLE IF NOT EXISTS model_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Timestamp TEXT,
            AUC REAL,
            Accuracy REAL
        )""")
        cur.execute("""CREATE TABLE IF NOT EXISTS retrain_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Timestamp TEXT,
            ModelType TEXT,
            Status TEXT
        )""")
        cur.execute("""CREATE TABLE IF NOT EXISTS team_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            team_name TEXT NOT NULL,
            conference TEXT NOT NULL,
            wins INTEGER NOT NULL,
            losses INTEGER NOT NULL,
            ties INTEGER NOT NULL,
            points_scored INTEGER NOT NULL,
            points_allowed INTEGER NOT NULL,
            win_percentage REAL NOT NULL,
            season INTEGER NOT NULL,
            UNIQUE(team_name, season)
        )""")
        # Indexes
        cur.execute("CREATE INDEX IF NOT EXISTS idx_picks_timestamp ON daily_picks(Timestamp)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_bankroll_timestamp ON bankroll_tracker(Timestamp)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_games_date ON nba_games(date)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_games_season ON nba_games(season)")

        init_schema_version(cur)
        con.commit()
    logging.info("✔ Database initialized")

# --- Logging helpers ---
def log_daily_pick(home_team, away_team, ev):
    with connect() as con:
        cur = con.cursor()
        cur.execute("""
            INSERT INTO daily_picks (Timestamp, home_team, away_team, ev)
            VALUES (?, ?, ?, ?)
        """, (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), home_team, away_team, ev))
        con.commit()

def update_bankroll(current_bankroll, roi):
    with connect() as con:
        cur = con.cursor()
        cur.execute("""
            INSERT INTO bankroll_tracker (Timestamp, CurrentBankroll, ROI)
            VALUES (?, ?, ?)
        """, (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), current_bankroll, roi))
        con.commit()

def record_model_metrics(auc, accuracy):
    with connect() as con:
        cur = con.cursor()
        cur.execute("""
            INSERT INTO model_metrics (Timestamp, AUC, Accuracy)
            VALUES (?, ?, ?)
        """, (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), auc, accuracy))
        con.commit()

def record_retrain(model_type, status):
    with connect() as con:
        cur = con.cursor()
        cur.execute("""
            INSERT INTO retrain_history (Timestamp, ModelType, Status)
            VALUES (?, ?, ?)
        """, (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), model_type, status))
        con.commit()

def get_bet_history(limit: int = 50):
    """Return bankroll history as a DataFrame."""
    with connect() as con:
        df = pd.read_sql(
            "SELECT * FROM bankroll_tracker ORDER BY Timestamp ASC LIMIT ?",
            con,
            params=(limit,)
        )
    return df