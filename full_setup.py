import os
import logging
import yaml
import sqlite3
import pandas as pd
from datetime import datetime

# === Local imports ===
from utils.nba_api import fetch_nba_games
from train.train_model_xgb import train_xgb_model
from utils.notify import send_message

# ======================
# Logging
# ======================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ======================
# Load config
# ======================
CONFIG = yaml.safe_load(open("config.yaml"))
DB_PATH = CONFIG["database"]["path"]
MODEL_DIR = CONFIG["model"]["models_dir"]

# ======================
# Create DB folder
# ======================
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ======================
# 1️⃣ CREATE DATABASE
# ======================
def create_database():
    logging.info("🗂 Creating database structure...")
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS nba_games (
        id INTEGER PRIMARY KEY,
        date TEXT,
        home_team TEXT,
        home_score INTEGER,
        visitor_team TEXT,
        visitor_score INTEGER
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
    CREATE TABLE IF NOT EXISTS bankroll_tracker (
        Timestamp TEXT,
        CurrentBankroll REAL,
        ROI REAL
    )
    """)

    con.commit()
    con.close()
    logging.info("✅ Database created successfully")


# ======================
# 2️⃣ FETCH & STORE GAMES
# ======================
def fetch_and_store_games(season:int = 2025):
    logging.info(f"📦 Fetching NBA games for season {season}...")
    df = fetch_nba_games(season)

    if df.empty:
        raise ValueError("❌ No games fetched!")

    df_db = df[['id', 'date', 'home_team', 'home_score', 'visitor_team', 'visitor_score']]

    con = sqlite3.connect(DB_PATH)
    df_db.to_sql("nba_games", con, if_exists="replace", index=False)
    con.close()

    logging.info(f"✔ Stored {len(df_db)} games into database")


# ======================
# 3️⃣ TRAIN MODEL
# ======================
def train_model():
    logging.info("🤖 Training XGBoost model...")
    train_xgb_model()  # uses train/train_model_xgb.py
    logging.info("🎯 Model training completed")


# ======================
# 4️⃣ SEND INITIAL TELEGRAM MESSAGE
# ======================
def send_initial_notification():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    msg = (
        "🚀 *NBA Analytics Pipeline Installed Successfully!*\n\n"
        f"🕒 Timestamp: {timestamp}\n"
        "📊 Database initialized\n"
        "📦 NBA game data loaded\n"
        "🤖 XGBoost model trained\n\n"
        "You're all set! 🎉"
    )
    send_message(msg)


# ======================
# 5️⃣ RUN EVERYTHING
# ======================
def run_full_setup():
    logging.info("=======================================")
    logging.info("🏀 Starting Full NBA Analytics Setup...")
    logging.info("=======================================")

    create_database()
    fetch_and_store_games(2025)
    train_model()
    send_initial_notification()

    logging.info("\n🎉 FULL SETUP COMPLETE!\n")
    logging.info("Next steps:")
    logging.info("1. Start Flask API →  python app.py")
    logging.info("2. Start daily scheduler → python scheduler/daily_picks.py")
    logging.info("3. Start Telegram bot → python telegram_bot/bot.py")
    logging.info("All systems are ready!\n")


if __name__ == "__main__":
    run_full_setup()
