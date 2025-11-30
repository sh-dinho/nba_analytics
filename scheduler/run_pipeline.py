import schedule
import time
import logging
from utils.fetch_games import store_games, fetch_nba_games
from train.train_model_xgb import train_xgb_model
from utils.notify import send_daily_picks
import sqlite3
import yaml
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../config.yaml")
CONFIG = yaml.safe_load(open(CONFIG_PATH))
DB_PATH = CONFIG["database"]["path"]
DAILY_TIME = CONFIG["notifications"]["daily_picks_time"]

def pipeline():
    logging.info("🚀 Running daily pipeline...")
    df = fetch_nba_games()
    if not df.empty:
        store_games(df)
        acc, auc = train_xgb_model()
        con = sqlite3.connect(DB_PATH)
        df_picks = pd.read_sql("SELECT * FROM nba_games LIMIT 5", con)
        con.close()
        send_daily_picks(df_picks)

schedule.every().day.at(DAILY_TIME).do(pipeline)

logging.info(f"📅 Scheduler started for daily {DAILY_TIME}")
while True:
    schedule.run_pending()
    time.sleep(60)
