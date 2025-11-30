import sqlite3
import yaml
import logging
import os
import pandas as pd

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
    con.row_factory = sqlite3.Row  # easier dict-like access
    return con

# --- Schema initialization ---
def init_db():
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    with connect() as con:
        cur = con.cursor()

        # Core bet tracking
        cur.execute("""
            CREATE TABLE IF NOT EXISTS bet_tracking (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bet_timestamp TEXT NOT NULL,
                bet_team TEXT NOT NULL,
                stake REAL NOT NULL,
                odds REAL NOT NULL,
                outcome TEXT CHECK(outcome IN ('win','loss')) NOT NULL,
                bankroll REAL NOT NULL,
                roi REAL,
                UNIQUE(bet_timestamp, bet_team)
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_bet_team ON bet_tracking(bet_team)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_bet_timestamp ON bet_tracking(bet_timestamp)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_outcome ON bet_tracking(outcome)")

        # Team performance
        cur.execute("""
            CREATE TABLE IF NOT EXISTS team_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                team_name TEXT NOT NULL,
                conference TEXT NOT NULL,
                wins INTEGER NOT NULL,
                losses INTEGER NOT NULL,
                points_scored INTEGER NOT NULL,
                points_allowed INTEGER NOT NULL,
                win_percentage REAL NOT NULL,
                season INTEGER NOT NULL,
                UNIQUE(team_name, season)
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_team_name ON team_performance(team_name)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_season ON team_performance(season)")

        # Raw games table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS nba_games (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_date TEXT,
                season INTEGER,
                home_team TEXT,
                away_team TEXT,
                home_score INTEGER,
                away_score INTEGER,
                winner TEXT
            )
        """)

        # Schema versioning
        cur.execute("""
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER NOT NULL,
                applied_on TEXT NOT NULL
            )
        """)
        cur.execute("SELECT COUNT(*) FROM schema_version")
        if cur.fetchone()[0] == 0:
            cur.execute("INSERT INTO schema_version (version, applied_on) VALUES (?, datetime('now'))", (1,))

        con.commit()
    logging.info("✔ Database initialized")

# --- Schema versioning ---
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

def migrate_to_v2():
    with connect() as con:
        cur = con.cursor()
        try:
            cur.execute("CREATE INDEX IF NOT EXISTS idx_bet_roi ON bet_tracking(roi)")
            con.commit()
            set_version(2)
            logging.info("✔ Migrated to schema version 2")
        except Exception as e:
            logging.error(f"❌ Migration to v2 failed: {e}")
            con.rollback()

def run_migrations():
    init_db()
    current_version = get_current_version()
    logging.info(f"Current schema version: {current_version}")
    if current_version < 2:
        migrate_to_v2()

# --- Bet history with filters ---
def get_bet_history(limit=20, team=None, start_date=None, end_date=None):
    query = """
        SELECT bet_timestamp, bet_team, stake, odds, outcome, bankroll, roi
        FROM bet_tracking WHERE 1=1
    """
    params = []
    if team:
        query += " AND bet_team = ?"
        params.append(team)
    if start_date:
        query += " AND bet_timestamp >= ?"
        params.append(start_date)
    if end_date:
        query += " AND bet_timestamp <= ?"
        params.append(end_date)
    query += " ORDER BY bet_timestamp DESC LIMIT ?"
    params.append(limit)

    with connect() as con:
        return pd.read_sql(query, con, params=params)

# --- Team performance updates ---
def update_team_performance(team_name, conference, wins, losses, points_scored, points_allowed, season):
    win_percentage = wins / (wins + losses) if (wins + losses) > 0 else 0
    with connect() as con:
        cur = con.cursor()
        cur.execute("""
            INSERT INTO team_performance (team_name, conference, wins, losses, points_scored, points_allowed, win_percentage, season)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(team_name, season) DO UPDATE SET
                wins=excluded.wins,
                losses=excluded.losses,
                points_scored=excluded.points_scored,
                points_allowed=excluded.points_allowed,
                win_percentage=excluded.win_percentage
        """, (team_name, conference, wins, losses, points_scored, points_allowed, win_percentage, season))
        con.commit()