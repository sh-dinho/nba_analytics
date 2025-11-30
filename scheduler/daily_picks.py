import schedule
import time
import pandas as pd
import sqlite3
from utils.notify import send_daily_picks
import logging

DB_PATH = "db/nba_analytics.db"
DAILY_TIME = "09:00"

def job():
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM daily_picks ORDER BY Timestamp DESC LIMIT 10", con)
    con.close()
    send_daily_picks(df)

schedule.every().day.at(DAILY_TIME).do(job)
logging.info(f"Scheduler started. Daily picks will be sent at {DAILY_TIME}")

while True:
    schedule.run_pending()
    time.sleep(60)
