# ============================================================
# Path: src/config_loader.py
# Filename: config_loader.py
# Author: Your Team
# Date: December 2025
# Purpose: Load and validate application configuration using
#          Pydantic Settings. Database fields are optional by
#          default and mapped to environment variables.
# Project: nba_analysis
# Version: 1.1 (adds dependencies section + clearer docstrings)
#
# Dependencies:
# - typing (standard library)
# - typing_extensions (for Literal fallback on Python <3.8)
# - pydantic
# - pydantic_settings
# ============================================================

from typing import Optional

try:
    # Python 3.8+ has Literal in typing
    from typing import Literal
except ImportError:
    # Fallback for Python <3.8
    from typing_extensions import Literal

from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings


class DatabaseSettings(BaseSettings):
    """
    Database configuration settings loaded from environment variables.
    All fields are optional by default.
    """
    host: Optional[str] = Field(default=None, env="DATABASE_HOST")
    port: Optional[int] = Field(default=None, env="DATABASE_PORT")
    user: Optional[str] = Field(default=None, env="DATABASE_USER")
    password: Optional[str] = Field(default=None, env="DATABASE_PASSWORD")


class LoggingSettings(BaseSettings):
    """
    Logging configuration settings.
    """
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    Includes NBA API key, database settings, and logging settings.
    """
    NBA_API_KEY: str
    database: DatabaseSettings = DatabaseSettings()
    logging: LoggingSettings = LoggingSettings()


def load_settings() -> Settings:
    """
    Load settings from environment variables.

    Returns:
        Settings: Application configuration object.

    Raises:
        ValidationError: If required values (e.g., NBA_API_KEY) are missing or invalid.
    """
    return Settings()
