# ============================================================
# File: notifications/__init__.py
# Purpose: Telegram helpers for text and photo sending
# ============================================================

import os
import requests
from nba_core.log_config import init_global_logger

logger = init_global_logger("notifications")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_message(text: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("⚠️ Telegram credentials not set; skipping message")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    resp = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text})
    if resp.status_code != 200:
        logger.warning(f"Telegram message failed: {resp.text}")

def send_photo(photo_path: str, caption: str = ""):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("⚠️ Telegram credentials not set; skipping photo")
        return
    if not os.path.exists(photo_path):
        logger.warning(f"⚠️ Photo path not found: {photo_path}")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    with open(photo_path, "rb") as photo:
        resp = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption}, files={"photo": photo})
    if resp.status_code != 200:
        logger.warning(f"Telegram photo failed: {resp.text}")
