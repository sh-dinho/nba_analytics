from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Unified Settings
# File: src/config/settings.py
# Author: Sadiq
# ============================================================

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from src.config import paths as PATHS
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
    # --------------------------------------------------------
    # Credentials
    # --------------------------------------------------------
    telegram_bot_token: str = TELEGRAM_BOT_TOKEN
    telegram_chat_id: str = TELEGRAM_CHAT_ID

    # Immutable API headers
    nba_api_headers: dict = field(
        default_factory=lambda: MappingProxyType(NBA_API_HEADERS)
    )

    odds_api_key: str = ODDS_API_KEY

    # --------------------------------------------------------
    # Feature Flags
    # --------------------------------------------------------
    enable_alerts: bool = FEATURE_FLAG_ENABLE_ALERTS
    enable_monitoring: bool = FEATURE_FLAG_ENABLE_MONITORING
    enable_backtesting: bool = FEATURE_FLAG_ENABLE_BACKTESTING

    # --------------------------------------------------------
    # Model Metadata
    # --------------------------------------------------------
    model_version: str = MODEL_VERSION
    model_environment: str = MODEL_ENVIRONMENT

    # --------------------------------------------------------
    # Paths (module reference)
    # --------------------------------------------------------
    paths: Any = field(default=PATHS, repr=False)


settings = Settings()
