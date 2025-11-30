import requests
import logging
import yaml
import sqlite3
import pandas as pd

CONFIG = yaml.safe_load(open("config.yaml"))
TELEGRAM_TOKEN = CONFIG["notifications"]["telegram_token"]
TELEGRAM_CHAT_ID = CONFIG["notifications"]["telegram_chat_id"]
DB_PATH = CONFIG["database"]["path"]

BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

def send_message(text):
    try:
        requests.post(f"{BASE_URL}/sendMessage", json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text
        })
        logging.info("✅ Telegram message sent")
    except Exception as e:
        logging.error(f"❌ Telegram send failed: {e}")

def send_daily_picks(df):
    if df.empty:
        send_message("❌ No picks today")
        return
    msg = "💰 Today's Picks 💰\n"
    for _, row in df.iterrows():
        msg += f"\n{row['home_team']} vs {row['visitor_team']} | Home Score: {row['home_score']} | Visitor Score: {row['visitor_score']}"
    send_message(msg)
