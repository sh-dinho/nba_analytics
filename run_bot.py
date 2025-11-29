import requests
import time
from utils.bot import handle_command, TELEGRAM_TOKEN

BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

def poll_updates():
    offset = None
    print("🚀 Telegram bot started")
    while True:
        try:
            resp = requests.get(f"{BASE_URL}/getUpdates", params={"timeout": 30, "offset": offset})
            updates = resp.json().get("result", [])
            for update in updates:
                offset = update["update_id"] + 1
                if "message" in update and "text" in update["message"]:
                    chat_id = update["message"]["chat"]["id"]
                    text = update["message"]["text"]
                    handle_command(chat_id, text)
        except Exception as e:
            print(f"Polling error: {e}")
        time.sleep(2)

if __name__ == "__main__":
    poll_updates()
