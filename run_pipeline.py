import schedule
import time
import sqlite3
import pandas as pd
import yaml
import logging
import os
from utils.nba_api import fetch_season_games
from train_model_xgb import MODEL_PATH
from utils.notify import send_daily_picks
import joblib
from datetime import datetime
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

CONFIG = yaml.safe_load(open("config.yaml"))
DB_PATH = CONFIG["database"]["path"]
DAILY_TIME = CONFIG["notifications"]["daily_picks_time"]

# ------------------- Pipeline -------------------
def run_pipeline():
    logging.info("🚀 Running daily pipeline...")

    # 1. Fetch latest NBA games
    season = datetime.now().year
    games_df = fetch_season_games(season)
    if games_df.empty:
        logging.warning("No games fetched")
        return

    # Store games in DB
    con = sqlite3.connect(DB_PATH)
    games_df.to_sql("nba_games", con, if_exists="append", index=False)
    con.close()
    logging.info(f"✔ Stored {len(games_df)} games")

    # 2. Load model
    if not os.path.exists(MODEL_PATH):
        logging.warning("Model not found. Training first...")
        import train_model_xgb
    model = joblib.load(MODEL_PATH)

    # 3. Feature engineering
    games_df["PTS_avg"] = games_df["AwayPTS"].rolling(5).mean()
    X = games_df[["PTS_avg"]].fillna(0)

    # 4. Predictions
    prob = model.predict_proba(X)[:,1]
    games_df["Probability"] = prob
    games_df["Odds"] = 2.0
    games_df["EV"] = (prob * 2.0) - 1
    games_df["SuggestedStake"] = np.maximum(0, games_df["EV"]) * CONFIG["betting"]["bankroll"]

    # 5. Filter EV picks
    picks_df = games_df[games_df["EV"] >= CONFIG["betting"]["min_ev_threshold"]].copy()
    if picks_df.empty:
        logging.info("No positive EV picks today")
    else:
        send_daily_picks(picks_df)

    # 6. Update bankroll
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT CurrentBankroll FROM bankroll_tracker ORDER BY Timestamp DESC LIMIT 1")
    row = cur.fetchone()
    starting_bankroll = row[0] if row else CONFIG["betting"]["bankroll"]
    profit = (picks_df["SuggestedStake"] * picks_df["EV"]).sum()
    new_bankroll = starting_bankroll + profit
    roi = (new_bankroll - starting_bankroll)/starting_bankroll if starting_bankroll>0 else 0
    cur.execute("""
        INSERT INTO bankroll_tracker (Timestamp, StartingBankroll, CurrentBankroll, ROI, Notes)
        VALUES (?, ?, ?, ?, ?)
    """, (datetime.now().strftime("%Y-%m-%d %H:%M"), starting_bankroll, new_bankroll, roi, "Daily update"))
    con.commit()
    con.close()
    logging.info(f"💰 Bankroll updated: {new_bankroll:.2f} (ROI={roi:.2%})")

# ------------------- Scheduler -------------------
def run_scheduler():
    schedule.every().day.at(DAILY_TIME).do(run_pipeline)
    logging.info(f"⏰ Scheduler set for {DAILY_TIME}")
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    run_scheduler()
