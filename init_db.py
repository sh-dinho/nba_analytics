import sqlite3
import yaml
import logging
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO)

# --- Safe config loading ---
try:
    with open("config.yaml") as f:
        CONFIG = yaml.safe_load(f)
except FileNotFoundError:
    logging.warning("⚠️ config.yaml not found, using defaults.")
    CONFIG = {"database": {"path": "bets.db"}}

DB_PATH = CONFIG["database"]["path"]

# --- Connection helper ---
def connect():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con

# --- Schema versioning ---
def init_schema_version(cur):
    cur.execute("""
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER NOT NULL,
            applied_on TEXT NOT NULL
        )
    """)
    cur.execute("SELECT COUNT(*) FROM schema_version")
    if cur.fetchone()[0] == 0:
        cur.execute("INSERT INTO schema_version (version, applied_on) VALUES (?, datetime('now'))", (1,))

def get_current_version():
    with connect() as con:
        cur = con.cursor()
        cur.execute("SELECT version FROM schema_version ORDER BY applied_on DESC LIMIT 1")
        row = cur.fetchone()
        return row[0] if row else 1

def set_version(new_version):
    with connect() as con:
        cur = con.cursor()
        cur.execute("INSERT INTO schema_version (version, applied_on) VALUES (?, datetime('now'))", (new_version,))
        con.commit()

# --- Schema initialization ---
def init_db():
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
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
        # Indexes
        cur.execute("CREATE INDEX IF NOT EXISTS idx_picks_timestamp ON daily_picks(Timestamp)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_bankroll_timestamp ON bankroll_tracker(Timestamp)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_games_date ON nba_games(date)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_games_season ON nba_games(season)")

        # Schema versioning
        init_schema_version(cur)

        con.commit()
    logging.info("✔ Database initialized")

# --- Helper functions for inserts ---
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

# --- Example runner ---
if __name__ == "__main__":
    init_db()
    log_daily_pick("Lakers", "Warriors", 0.12)
    update_bankroll(150.0, 0.05)
    record_model_metrics(0.82, 0.74)
    record_retrain("Classifier", "Success")
    logging.info("✔ Sample records inserted")