import schedule
import time
import subprocess
import logging
import sqlite3
from utils.notify import send_daily_picks
import yaml

CONFIG = yaml.safe_load(open("config.yaml"))
DB_PATH = CONFIG["database"]["path"]
DAILY_TIME = CONFIG["notifications"]["daily_picks_time"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def job_fetch_and_train():
    logging.info("🚀 Running daily fetch & train job")
    try:
        subprocess.run(["python", "fetch_games.py"], check=True)
        subprocess.run(["python", "train_model_xgb.py"], check=True)
        con = sqlite3.connect(DB_PATH)
        df = pd.read_sql("SELECT * FROM nba_games ORDER BY date DESC LIMIT 10", con)
        con.close()
        send_daily_picks(df)
    except Exception as e:
        logging.error(f"❌ Daily job failed: {e}")

def run_scheduler():
    schedule.every().day.at(DAILY_TIME).do(job_fetch_and_train)
    logging.info(f"🕒 Scheduler started, daily at {DAILY_TIME}")
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    run_scheduler()
