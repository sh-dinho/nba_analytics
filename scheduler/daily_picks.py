import schedule
import time
import sqlite3
import pandas as pd
from utils.notify import send_daily_picks
import yaml
import logging

logging.basicConfig(level=logging.INFO)

CONFIG = yaml.safe_load(open("config.yaml"))
DB_PATH = CONFIG["database"]["path"]
DAILY_TIME = CONFIG["notifications"]["daily_picks_time"]

def job():
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM nba_games ORDER BY date DESC LIMIT 10", con)
    con.close()
    send_daily_picks(df)

schedule.every().day.at(DAILY_TIME).do(job)

logging.info(f"🚀 Daily picks scheduler started at {DAILY_TIME}")
while True:
    schedule.run_pending()
    time.sleep(60)
