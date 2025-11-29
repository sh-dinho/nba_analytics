import requests
import logging
import sqlite3
import pandas as pd
import yaml

CONFIG = yaml.safe_load(open("config.yaml"))
DB_PATH = CONFIG["database"]["path"]
TELEGRAM_TOKEN = CONFIG["notifications"]["telegram_token"]
TELEGRAM_CHAT_ID = CONFIG["notifications"]["telegram_chat_id"]
BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

def send_telegram_message(text):
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
    try:
        requests.post(f"{BASE_URL}/sendMessage", json=payload, timeout=10)
        logging.info("Telegram message sent")
    except Exception as e:
        logging.error(f"Failed to send Telegram message: {e}")

def send_daily_picks(df):
    if df.empty:
        send_telegram_message("❌ No picks available today.")
        return
    msg = "💰 Today's Picks:\n"
    for _, row in df.iterrows():
        msg += f"{row['Visitor/Neutral']} @ {row['Home/Neutral']} | Odds {row.get('Odds', 'N/A')} | EV {row.get('EV', 'N/A')}\n"
    send_telegram_message(msg)
