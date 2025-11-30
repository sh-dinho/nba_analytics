import os
import time
import logging
import schedule
import sqlite3
import pandas as pd
from datetime import datetime
import yaml
from utils.fetch_games import init_db, fetch_nba_games, store_games
from train.train_model_xgb import train_xgb_model
from utils.notify import send_daily_picks

# ----------------------------
# CONFIGURATION
# ----------------------------
ROOT = os.path.dirname(os.path.dirname(__file__))
CONFIG_PATH = os.path.join(ROOT, "config.yaml")
CONFIG = yaml.safe_load(open(CONFIG_PATH))

DB_PATH = CONFIG["database"]["path"]
DAILY_TIME = CONFIG["notifications"]["daily_picks_time"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# ----------------------------
# JOB FUNCTIONS
# ----------------------------
def job_fetch_games():
    logging.info("📥 Running job: Fetch NBA games")
    init_db()
    df_games = fetch_nba_games()
    if df_games.empty:
        logging.error("❌ No games fetched today")
    else:
        store_games(df_games)


def job_retrain_model():
    logging.info("🧠 Running job: Retrain XGBoost model")
    try:
        train_xgb_model()
        logging.info("✔ Model retrained successfully")
    except Exception as e:
        logging.error(f"❌ Model retrain failed: {e}")


def job_send_daily_picks():
    logging.info("💰 Running job: Send daily picks")
    try:
        con = sqlite3.connect(DB_PATH)
        df = pd.read_sql("SELECT * FROM daily_picks ORDER BY Timestamp DESC LIMIT 10", con)
        con.close()
        send_daily_picks(df)
    except Exception as e:
        logging.error(f"❌ Failed to send daily picks: {e}")


# ----------------------------
# SCHEDULER SETUP
# ----------------------------
schedule.every().day.at("08:00").do(job_fetch_games)      # Fetch NBA games every morning
schedule.every().day.at("09:00").do(job_retrain_model)   # Retrain model after fetching
schedule.every().day.at(DAILY_TIME).do(job_send_daily_picks)  # Send picks at configured time

logging.info(f"🚀 Scheduler started. Daily picks will be sent at {DAILY_TIME}.")

# ----------------------------
# RUN LOOP
# ----------------------------
if __name__ == "__main__":
    while True:
        schedule.run_pending()
        time.sleep(30)
