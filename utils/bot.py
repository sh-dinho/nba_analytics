import requests
import sqlite3
import pandas as pd
import logging
import yaml
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

CONFIG = yaml.safe_load(open("config.yaml"))
TELEGRAM_TOKEN = CONFIG["notifications"]["telegram_token"]
AUTHORIZED_CHAT_ID = int(CONFIG["notifications"]["telegram_chat_id"])
API_URL = CONFIG["server"]["api_url"]
DB_PATH = CONFIG["database"]["path"]
BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

def send_message(chat_id: int, text: str):
    try:
        requests.post(f"{BASE_URL}/sendMessage", json={"chat_id": chat_id, "text": text})
    except Exception as e:
        logging.error(f"Failed to send message: {e}")

def handle_command(chat_id: int, command: str):
    if chat_id != AUTHORIZED_CHAT_ID:
        send_message(chat_id, "❌ You are not authorized.")
        return

    command = command.lower()
    if command == "/status":
        try:
            con = sqlite3.connect(DB_PATH)
            df = pd.read_sql("SELECT * FROM retrain_history ORDER BY Timestamp DESC LIMIT 1", con)
            con.close()
            if not df.empty:
                last = df.iloc[0]
                send_message(chat_id, f"📊 Last retrain: {last['Timestamp']} ({last['ModelType']}, {last['Status']})")
            else:
                send_message(chat_id, "⚠ No retrain history found.")
        except Exception as e:
            send_message(chat_id, f"❌ Error fetching status: {e}")

    elif command == "/picks":
        try:
            con = sqlite3.connect(DB_PATH)
            df = pd.read_sql("SELECT * FROM daily_picks ORDER BY Timestamp DESC LIMIT 5", con)
            con.close()
            if not df.empty:
                msg = "💰 Latest Picks:\n"
                for _, row in df.iterrows():
                    msg += f"{row['Team']} vs {row['Opponent']} | Odds {row['Odds']} | Prob {row['Probability']:.2f} | EV {row['EV']:.3f}\n"
                send_message(chat_id, msg)
            else:
                send_message(chat_id, "❌ No picks available.")
        except Exception as e:
            send_message(chat_id, f"❌ Error fetching picks: {e}")

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

    elif command == "/metrics":
        try:
            con = sqlite3.connect(DB_PATH)
            df = pd.read_sql("SELECT * FROM model_metrics ORDER BY Timestamp DESC LIMIT 1", con)
            con.close()
            if not df.empty:
                row = df.iloc[0]
                send_message(chat_id, f"📊 Model Metrics:\nAccuracy: {row['Accuracy']:.3f}\nAUC: {row['AUC']:.3f}")
            else:
                send_message(chat_id, "⚠ No metrics found.")
        except Exception as e:
            send_message(chat_id, f"❌ Error fetching metrics: {e}")

    else:
        send_message(chat_id, "🤖 Unknown command. Try /status, /picks, /retrain, /metrics")
