import os
import sqlite3
import logging
from datetime import datetime
import pandas as pd

# Import utility functions
from utils.nba_api import fetch_nba_games
from train.train_model_xgb import train_xgb_model
from utils.notify import send_daily_picks

# --- Config ---
DB_PATH = "db/nba_analytics.db"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_model.pkl")
os.makedirs("db", exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# --- 1️⃣ Create tables ---
con = sqlite3.connect(DB_PATH)
cur = con.cursor()

logging.info("Creating database tables if not exist...")
cur.execute("""
CREATE TABLE IF NOT EXISTS nba_games (
    GameID TEXT PRIMARY KEY,
    Date TEXT,
    Visitor TEXT,
    Visitor_PTS INTEGER,
    Home TEXT,
    Home_PTS INTEGER
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
    Timestamp TEXT,
    Team TEXT,
    Opponent TEXT,
    Odds REAL,
    Probability REAL,
    EV REAL,
    SuggestedStake REAL
)
""")
con.commit()
con.close()
logging.info("✅ Database setup complete.")

# --- 2️⃣ Fetch NBA games ---
logging.info("Fetching NBA games...")
df_games = fetch_nba_games()
if df_games.empty:
    logging.error("❌ No games fetched. Exiting setup.")
    exit()

# --- 3️⃣ Store games ---
logging.info("Storing NBA games into database...")
con = sqlite3.connect(DB_PATH)
for _, row in df_games.iterrows():
    con.execute("""
    INSERT OR REPLACE INTO nba_games (GameID, Date, Visitor, Visitor_PTS, Home, Home_PTS)
    VALUES (?,
