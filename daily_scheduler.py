import schedule
import time
import sqlite3
import pandas as pd
import logging
import yaml
from utils.notify import send_daily_picks
from train_model_xgb import train_model
from fetch_games import main as fetch_games_main

logging.basicConfig(level=logging.INFO)

CONFIG = yaml.safe_load(open("config.yaml"))
DB_PATH = CONFIG["database"]["path"]
DAILY_TIME = CONFIG["notifications"]["daily_picks_time"]

def job():
    fetch_games_main()
    train_model()
    conn = sqlite3.connect(DB_PATH)
    df_picks = pd.read_sql("SELECT * FROM nba_games ORDER BY Timestamp DESC LIMIT 10", conn)
    conn.close()
    send_daily_picks(df_picks)

schedule.every().day.at(DAILY_TIME).do(job)
logging.info(f"Scheduler started | Daily at {DAILY_TIME}")

while True:
    schedule.run_pending()
    time.sleep(60)
