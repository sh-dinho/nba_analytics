# ============================================================
# Path: src/config_loader.py
# Filename: config_loader.py
# Author: Your Team
# Date: December 2025
# Purpose: Load and validate application configuration using
#          Pydantic Settings. Database fields are optional by
#          default and mapped to environment variables.
# ============================================================

from typing import Optional
try:
    # Python 3.8+ has Literal in typing
    from typing import Literal
except ImportError:
    # Fallback for Python <3.8
    from typing_extensions import Literal

from pydantic_settings import BaseSettings
from pydantic import Field, ValidationError

class DatabaseSettings(BaseSettings):
    host: Optional[str] = Field(default=None, env="DATABASE_HOST")
    port: Optional[int] = Field(default=None, env="DATABASE_PORT")
    user: Optional[str] = Field(default=None, env="DATABASE_USER")
    password: Optional[str] = Field(default=None, env="DATABASE_PASSWORD")

class LoggingSettings(BaseSettings):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

class Settings(BaseSettings):
    NBA_API_KEY: str
    database: DatabaseSettings = DatabaseSettings()
    logging: LoggingSettings = LoggingSettings()

def load_settings() -> Settings:
    """
    Load settings from environment variables.
    Raises ValidationError if required values are missing.
    """
    return Settings()
