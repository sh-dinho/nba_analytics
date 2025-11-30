import logging
import requests
from pathlib import Path
import yaml
import sqlite3
import pandas as pd
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = Path(__file__).parent.parent.resolve()
CONFIG_PATH = BASE_DIR / "config.yaml"
DB_PATH = BASE_DIR / "nba_analytics.db"

if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Config file not found at {CONFIG_PATH}")
CONFIG = yaml.safe_load(open(CONFIG_PATH))

API_URL = CONFIG["server"]["api_url"]
TELEGRAM_TOKEN = CONFIG["notifications"]["telegram_token"]
AUTHORIZED_CHAT_ID = int(CONFIG["notifications"]["telegram_chat_id"])
BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

def send_message(chat_id: int, text: str):
    url = f"{BASE_URL}/sendMessage"
    requests.post(url, json={"chat_id": chat_id, "text": text})

def handle_command(chat_id: int, command: str):
    if chat_id != AUTHORIZED_CHAT_ID:
        send_message(chat_id, "❌ Unauthorized")
        return
    if command == "/status":
        con = sqlite3.connect(DB_PATH)
        df = pd.read_sql("SELECT * FROM retrain_history ORDER BY Timestamp DESC LIMIT 1", con)
        con.close()
        if not df.empty:
            last = df.iloc[0]
            send_message(chat_id, f"📊 Last retrain: {last['Timestamp']}")
        else:
            send_message(chat_id, "⚠ No retrain history found.")
    elif command == "/picks":
        con = sqlite3.connect(DB_PATH)
        df = pd.read_sql("SELECT * FROM daily_picks ORDER BY Timestamp DESC LIMIT 5", con)
        con.close()
        if not df.empty:
            msg = "💰 Latest Picks:\n"
            for _, row in df.iterrows():
                msg += f"\n{row['Team']} vs {row['Opponent']} | Odds {row['Odds']} | Prob {row['Probability']:.2f} | EV {row['EV']:.3f}"
            send_message(chat_id, msg)
        else:
            send_message(chat_id, "❌ No picks available.")
    elif command == "/retrain":
        try:
            resp = requests.post(f"{API_URL}/train/run")
            data = resp.json()
            if data.get("status") == "ok":
                send_message(chat_id, "🔄 Retrain started successfully.")
            else:
                send_message(chat_id, f"❌ Retrain failed: {data.get('message')}")
        except Exception as e:
            send_message(chat_id, f"❌ Error triggering retrain: {e}")
    else:
        send_message(chat_id, "🤖 Unknown command")

def poll_updates():
    offset = None
    logging.info("🚀 Telegram bot started")
    while True:
        try:
            url = f"{BASE_URL}/getUpdates"
            params = {"timeout": 30, "offset": offset}
            resp = requests.get(url, params=params).json()
            for update in resp.get("result", []):
                offset = update["update_id"] + 1
                if "message" in update and "text" in update["message"]:
                    chat_id = update["message"]["chat"]["id"]
                    command = update["message"]["text"].strip()
                    handle_command(chat_id, command)
        except Exception as e:
            logging.error(f"Polling error: {e}")
        time.sleep(2)

if __name__ == "__main__":
    poll_updates()
