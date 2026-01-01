from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Environment Configuration
# File: src/config/env.py
# Author: Sadiq
#
# Description:
#     Centralized environment variable loader for:
#       • Telegram alerts
#       • NBA API headers
#       • Odds API keys
#       • Model registry metadata
#       • Feature flags
# ============================================================

import os
from loguru import logger

# ------------------------------------------------------------
# Telegram
# ------------------------------------------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    logger.warning("Telegram credentials missing — alerts may be disabled.")

# ------------------------------------------------------------
# NBA API Headers
# ------------------------------------------------------------
NBA_API_HEADERS = {
    "Host": "stats.nba.com",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "Connection": "keep-alive",
}

# ------------------------------------------------------------
# Odds API Keys
# ------------------------------------------------------------
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
if not ODDS_API_KEY:
    logger.warning("ODDS_API_KEY not set — odds ingestion may fail.")

# ------------------------------------------------------------
# Feature Flags
# ------------------------------------------------------------
FEATURE_FLAG_ENABLE_ALERTS = os.getenv("ENABLE_ALERTS", "1") == "1"
FEATURE_FLAG_ENABLE_MONITORING = os.getenv("ENABLE_MONITORING", "1") == "1"
FEATURE_FLAG_ENABLE_BACKTESTING = os.getenv("ENABLE_BACKTESTING", "1") == "1"

# ------------------------------------------------------------
# Model Metadata
# ------------------------------------------------------------
MODEL_VERSION = os.getenv("MODEL_VERSION", "default")
MODEL_ENVIRONMENT = os.getenv("MODEL_ENVIRONMENT", "production")