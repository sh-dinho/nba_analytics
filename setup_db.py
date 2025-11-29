import sqlite3
import yaml
import os

# Load config
CONFIG_PATH = "config.yaml"
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Config file not found at {CONFIG_PATH}")

CONFIG = yaml.safe_load(open(CONFIG_PATH))
DB_PATH = CONFIG["database"]["path"]

# Connect to SQLite
con = sqlite3.connect(DB_PATH)
cur = con.cursor()

# ------------------- nba_games -------------------
cur.execute("""
CREATE TABLE IF NOT EXISTS nba_games (
    GAME_ID TEXT PRIMARY KEY,
    GAME_DATE TEXT,
    TEAM_ABBREVIATION TEXT,
    PTS REAL,
    REB REAL,
    AST REAL,
    WL TEXT,
    MATCHUP TEXT
);
""")

# ------------------- daily_picks -------------------
cur.execute("""
CREATE TABLE IF NOT EXISTS daily_picks (
    Timestamp TEXT,
    Team TEXT,
    Opponent TEXT,
    Probability REAL,
    Odds REAL,
    EV REAL,
    SuggestedStake REAL
);
""")

# ------------------- bankroll_tracker -------------------
cur.execute("""
CREATE TABLE IF NOT EXISTS bankroll_tracker (
    Timestamp TEXT,
    StartingBankroll REAL,
    CurrentBankroll REAL,
    ROI REAL,
    Notes TEXT
);
""")

# ------------------- model_metrics -------------------
cur.execute("""
CREATE TABLE IF NOT EXISTS model_metrics (
    Timestamp TEXT,
    ModelType TEXT,
    Accuracy REAL,
    AUC REAL
);
""")

# ------------------- retrain_history -------------------
cur.execute("""
CREATE TABLE IF NOT EXISTS retrain_history (
    Timestamp TEXT,
    ModelType TEXT,
    Status TEXT
);
""")

con.commit()
con.close()
print(f"✅ Database initialized successfully at {DB_PATH}")
