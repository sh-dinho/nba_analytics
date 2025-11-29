import schedule
import time
from scripts.run_pipeline import run_pipeline
import logging
import yaml

CONFIG = yaml.safe_load(open("config.yaml"))
DAILY_TIME = CONFIG["notifications"]["daily_picks_time"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def job():
    logging.info("🚀 Running scheduled pipeline...")
    run_pipeline()

schedule.every().day.at(DAILY_TIME).do(job)
logging.info(f"📅 Scheduler set for {DAILY_TIME} daily")

while True:
    schedule.run_pending()
    time.sleep(60)
