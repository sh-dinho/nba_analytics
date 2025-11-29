import requests
import logging
import sqlite3
import pandas as pd
from configparser import ConfigParser

CONFIG_PATH = "config.yaml"
import yaml
CONFIG = yaml.safe_load(open(CONFIG_PATH))
TELEGRAM_TOKEN = CONFIG["notifications"]["telegram_token"]
TELEGRAM_CHAT_ID = CONFIG["notifications"]["telegram_chat_id"]
DB_PATH = CONFIG["database"]["path"]

BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

def send_telegram_message(text: str):
    try:
        url = f"{BASE_URL}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
        resp = requests.post(url, json=payload)
        resp.raise_for_status()
        logging.info("✅ Telegram message sent")
    except Exception as e:
        logging.error(f"❌ Failed to send Telegram message: {e}")

def send_daily_picks(df: pd.DataFrame):
    if df.empty:
        send_telegram_message("❌ No picks available today.")
        return
    msg = "🏀 Today's Picks:\n"
    for _, row in df.iterrows():
        msg += f"{row['HOME_TEAM']} vs {row['VISITOR_TEAM']} | Score: {row['HOME_SCORE']}-{row['VISITOR_SCORE']}\n"
    send_telegram_message(msg)
