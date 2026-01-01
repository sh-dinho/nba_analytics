from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Unified Settings
# File: src/config/settings.py
# Author: Sadiq
# ============================================================

from dataclasses import dataclass
from src.config import paths
from src.config.env import (
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
    NBA_API_HEADERS,
    ODDS_API_KEY,
    FEATURE_FLAG_ENABLE_ALERTS,
    FEATURE_FLAG_ENABLE_MONITORING,
    FEATURE_FLAG_ENABLE_BACKTESTING,
    MODEL_VERSION,
    MODEL_ENVIRONMENT,
)

@dataclass(frozen=True)
class Settings:
    telegram_bot_token: str = TELEGRAM_BOT_TOKEN
    telegram_chat_id: str = TELEGRAM_CHAT_ID
    nba_api_headers: dict = NBA_API_HEADERS
    odds_api_key: str = ODDS_API_KEY

    enable_alerts: bool = FEATURE_FLAG_ENABLE_ALERTS
    enable_monitoring: bool = FEATURE_FLAG_ENABLE_MONITORING
    enable_backtesting: bool = FEATURE_FLAG_ENABLE_BACKTESTING

    model_version: str = MODEL_VERSION
    model_environment: str = MODEL_ENVIRONMENT

    paths = paths

settings = Settings()