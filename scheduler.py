# scheduler.py
import sqlite3
import requests
import pandas as pd
import yaml
import logging
import os
from apscheduler.schedulers.blocking import BlockingScheduler
from datetime import datetime
from utils import send_telegram_message

logging.basicConfig(level=logging.INFO)

def load_config(path="config.yaml"):
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise RuntimeError("⚠️ config.yaml not found. Please create one with database.path and server.api_url")

CONFIG = load_config()
DB_PATH = CONFIG["database"]["path"]
API_URL = CONFIG["server"]["api_url"]

def connect():
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con

def fetch_new_games(season=2025):
    all_games = []
    page = 1
    while True:
        try:
            params = {"seasons[]": season, "per_page": 100, "page": page}
            r = requests.get(f"{API_URL}/games", params=params, timeout=15)
            r.raise_for_status()
        except requests.RequestException as e:
            logging.error(f"❌ Failed to fetch games: {e}")
            break

        data = r.json()
        games = data.get("data", [])
        if not games:
            break

        for g in games:
            home_score, away_score = g["home_team_score"], g["visitor_team_score"]
            winner = (
                g["home_team"]["full_name"] if home_score > away_score
                else g["visitor_team"]["full_name"] if away_score > home_score
                else "Tie"
            )
            all_games.append({
                "game_id": g["id"],
                "date": g["date"],
                "season": g["season"],
                "home_team": g["home_team"]["full_name"],
                "away_team": g["visitor_team"]["full_name"],
                "home_score": home_score,
                "away_score": away_score,
                "winner": winner
            })

        if page >= data.get("meta", {}).get("total_pages", 1):
            break
        page += 1

    return pd.DataFrame(all_games)

def store_games(df):
    if df.empty:
        logging.info("No new games to store.")
        return
    with connect() as con:
        cur = con.cursor()
        cur.executemany("""
            INSERT OR IGNORE INTO nba_games
            (game_id,date,season,home_team,away_team,home_score,away_score,winner)
            VALUES (?,?,?,?,?,?,?,?)
        """, df[["game_id","date","season","home_team","away_team","home_score","away_score","winner"]].values.tolist())
        con.commit()
    logging.info(f"✔ Stored {len(df)} games")

def send_daily_report():
    logging.info("Running daily job...")
    df = fetch_new_games(season=2025)
    store_games(df)

    with connect() as con:
        tracker = pd.read_sql("SELECT * FROM bankroll_tracker ORDER BY Timestamp DESC LIMIT 1", con)

    if tracker.empty:
        msg = f"📊 Daily Report ({datetime.now().strftime('%Y-%m-%d')})\nNo bankroll records yet.\nGames fetched: {len(df)}"
    else:
        latest = tracker.iloc[0]
        msg = (f"📊 Daily Report ({datetime.now().strftime('%Y-%m-%d')})\n"
               f"Bankroll: ${latest['CurrentBankroll']:.2f}\n"
               f"ROI: {latest['ROI']:.2%}\n"
               f"Games fetched: {len(df)}")
    try:
        send_telegram_message(msg)
        logging.info("✔ Report sent")
    except Exception as e:
        logging.error(f"❌ Failed to send Telegram report: {e}")

if __name__ == "__main__":
    scheduler = BlockingScheduler()
    scheduler.add_job(send_daily_report, "cron", hour=2, minute=0)  # 2 AM local
    logging.info("Scheduler started. Waiting for jobs...")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logging.info("Scheduler stopped.")