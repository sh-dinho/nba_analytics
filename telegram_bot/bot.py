import requests
import yaml
import logging
import sqlite3
import pandas as pd
import time
import os

# ======================
# Logging
# ======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# ======================
# Load config
# ======================
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
CONFIG = yaml.safe_load(open(CONFIG_PATH))

DB_PATH = CONFIG["database"]["path"]
TELEGRAM_TOKEN = CONFIG["notifications"]["telegram_token"]
AUTHORIZED_CHAT_ID = int(CONFIG["notifications"]["telegram_chat_id"])
API_URL = CONFIG["server"]["api_url"]

BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"


# ======================
# TELEGRAM SEND MESSAGE
# ======================
def send_message(chat_id: int, text: str):
    """Send a plain text message."""
    url = f"{BASE_URL}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    try:
        requests.post(url, json=payload)
    except Exception as e:
        logging.error(f"Failed to send message: {e}")


# ======================
# HANDLE COMMANDS
# ======================
def handle_command(chat_id: int, command: str):
    # Secure: Only allow your Telegram account
    if chat_id != AUTHORIZED_CHAT_ID:
        logging.warning(f"Unauthorized user blocked → Chat ID: {chat_id}")
        send_message(chat_id, "🚫 You are not authorized to use this bot.")
        return

    # -------------------------------------------
    # /status → latest retrain record
    # -------------------------------------------
    if command == "/status":
        con = sqlite3.connect(DB_PATH)
        df = pd.read_sql("SELECT * FROM retrain_history ORDER BY Timestamp DESC LIMIT 1", con)
        con.close()

        if df.empty:
            send_message(chat_id, "⚠ No retraining history found.")
        else:
            last = df.iloc[0]
            msg = (
                "📊 *Model Status*\n\n"
                f"🕒 Last Retrain: {last['Timestamp']}\n"
                f"🤖 Model: {last['ModelType']}\n"
                f"📌 Status: {last['Status']}"
            )
            send_message(chat_id, msg)

    # -------------------------------------------
    # /picks → show last 5 model picks
    # -------------------------------------------
    elif command == "/picks":
        con = sqlite3.connect(DB_PATH)
        df = pd.read_sql("SELECT * FROM daily_picks ORDER BY Timestamp DESC LIMIT 5", con)
        con.close()

        if df.empty:
            send_message(chat_id, "❌ No picks available yet.")
            return

        text = "🏀 *Latest Picks*\n"
        for _, row in df.iterrows():
            text += (
                f"\n\n🔹 {row['Team']} vs {row['Opponent']}"
                f"\n📈 Probability: {row['Probability']:.2f}"
                f"\n💰 Odds: {row['Odds']}"
                f"\n📊 EV: {row['EV']:.3f}"
            )

        send_message(chat_id, text)

    # -------------------------------------------
    # /retrain → trigger training pipeline
    # -------------------------------------------
    elif command == "/retrain":
        try:
            response = requests.post(f"{API_URL}/train/run", timeout=10)
            data = response.json()

            if data.get("status") == "ok":
                send_message(chat_id, "🔄 Model retraining has started...")
            else:
                send_message(chat_id, f"❌ Retrain failed: {data.get('message')}")

        except Exception as e:
            send_message(chat_id, f"❌ Error contacting API: {str(e)}")

    # -------------------------------------------
    # Unknown command
    # -------------------------------------------
    else:
        send_message(chat_id, "🤖 Unknown command.\nTry: /status /picks /retrain")


# ======================
# POLLING LOOP
# ======================
def poll_updates():
    offset = None
    logging.info("🤖 Telegram Bot started. Waiting for commands...")

    while True:
        try:
            url = f"{BASE_URL}/getUpdates"
            params = {"timeout": 30, "offset": offset}
            response = requests.get(url, params=params, timeout=40).json()

            for update in response.get("result", []):
                offset = update["update_id"] + 1

                if "message" in update and "text" in update["message"]:
                    chat_id = update["message"]["chat"]["id"]
                    command = update["message"]["text"].strip()

                    logging.info(f"📥 Received command: {command} from {chat_id}")
                    handle_command(chat_id, command)

        except Exception as e:
            logging.error(f"Polling error: {e}")

        time.sleep(2)  # prevent flooding


# ======================
# MAIN ENTRY
# ======================
if __name__ == "__main__":
    poll_updates()
