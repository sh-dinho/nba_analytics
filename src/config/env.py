# ============================================================
# Project: NBA Analytics & Betting Engine
# Author: Sadiq
# Description: Environment variables and secrets.
# ============================================================

import os

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

NBA_API_HEADERS = {
    "User-Agent": os.getenv("NBA_USER_AGENT", "Mozilla/5.0"),
}
