import requests
import time
import yaml
import sqlite3
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

CONFIG = yaml.safe_load(open("../config.yaml"))
TOKEN = CONFIG["notifications"]["telegram_token"]
CHAT_ID = CONFIG["notifications"]["telegram_chat_id"]
API_URL = CONFIG["server"]["api_url"]

BASE_URL = f"https://api.telegram.org/bot{TOKEN}"

def send_message(text):
    try:
        requests.post(f"{BASE_URL}/sendMessage", json={"chat_id": CHAT_ID, "text": text})
    except Exception as e:
        logging.error(f"Failed to send Telegram message: {e}")

def poll_updates():
    offset = None
    while True:
        try:
            r = requests.get(f"{BASE_URL}/getUpdates", params={"timeout": 30, "offset": offset}).json()
            for u in r.get("result", []):
                offset = u["update_id"] + 1
                if "message" in u and "text" in u["message"]:
                    cmd = u["message"]["text"]
                    if cmd == "/status":
                        send_message("Bot is running ✅")
                    elif cmd == "/retrain":
                        requests.post(f"{API_URL}/train/run")
                        send_message("Retraining triggered 🔄")
                    else:
                        send_message("Unknown command 🤖")
        except Exception as e:
            logging.error(e)
        time.sleep(2)

if __name__ == "__main__":
    poll_updates()
