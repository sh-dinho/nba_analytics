# ============================================================
# File: scripts/test_telegram.py
# Purpose: Send a test message to Telegram using TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID
# ============================================================

import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
    raise ValueError("❌ TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set in environment")

def send_test_message():
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": "✅ Hello from NBA Analytics pipeline test!",
        "parse_mode": "Markdown"
    }

    try:
        resp = requests.post(url, json=payload, timeout=10)
        print("Status:", resp.status_code)
        print("Response:", resp.json())
    except Exception as e:
        print("❌ Error sending Telegram message:", e)

if __name__ == "__main__":
    send_test_message()
