from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Environment Configuration
# File: src/config/env.py
# Author: Sadiq
# ============================================================

import os
from loguru import logger


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _get_env(name: str, default: str = "") -> str:
    """Load and normalize environment variables."""
    return os.getenv(name, default).strip()


def _get_flag(name: str, default: bool = True) -> bool:
    """Parse boolean feature flags safely."""
    raw = os.getenv(name, "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return default


# ------------------------------------------------------------
# Telegram Alerts
# ------------------------------------------------------------
TELEGRAM_BOT_TOKEN = _get_env("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = _get_env("TELEGRAM_CHAT_ID")

FEATURE_FLAG_ENABLE_ALERTS = _get_flag("ENABLE_ALERTS", True)

if FEATURE_FLAG_ENABLE_ALERTS:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Alerts enabled but Telegram credentials are missing.")
else:
    logger.info("Alerts disabled via feature flag.")


# ------------------------------------------------------------
# NBA API Headers (immutable)
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
# Odds API
# ------------------------------------------------------------
ODDS_API_KEY = _get_env("ODDS_API_KEY")
FEATURE_FLAG_ENABLE_MONITORING = _get_flag("ENABLE_MONITORING", True)

if FEATURE_FLAG_ENABLE_MONITORING and not ODDS_API_KEY:
    logger.warning("Monitoring enabled but ODDS_API_KEY is missing.")


# ------------------------------------------------------------
# Backtesting Feature Flag
# ------------------------------------------------------------
FEATURE_FLAG_ENABLE_BACKTESTING = _get_flag("ENABLE_BACKTESTING", True)


# ------------------------------------------------------------
# Model Metadata
# ------------------------------------------------------------
MODEL_VERSION = _get_env("MODEL_VERSION", "default")
MODEL_ENVIRONMENT = _get_env("MODEL_ENVIRONMENT", "production")
