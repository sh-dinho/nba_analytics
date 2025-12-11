# ============================================================
# File: src/utils/__init__.py
# Purpose: Consolidated exports for all utility modules
# Project: nba_analysis
# Version: 1.1 (fixed stale NBA API exports)
# ============================================================

# Centralized imports from all utility modules

# Logging configuration
from .logging_config import configure_logging

# I/O utilities
from .io import load_dataframe, save_dataframe, read_or_create

# Data cleaning
from .data_cleaning import clean_data, rename_columns, prepare_game_data

# Unique ID generation
from .add_unique_id import add_unique_id

# Mapping helpers
from .mapping import map_team_ids, map_player_ids, map_ids

# NBA API wrapper
from src.api.nba_api_client import fetch_season_games, fetch_boxscores

# Validation helpers
from .validation import (
    validate_game_ids,
    validate_season,
    validate_file_extension,
    validate_inputs,
)

__all__ = [
    # Logging
    "configure_logging",
    # I/O
    "load_dataframe",
    "save_dataframe",
    "read_or_create",
    # Cleaning
    "clean_data",
    "rename_columns",
    "prepare_game_data",
    # Unique IDs
    "add_unique_id",
    # Mapping
    "map_team_ids",
    "map_player_ids",
    "map_ids",
    # NBA API
    "fetch_season_games",
    "fetch_boxscores",
    # Validation
    "validate_game_ids",
    "validate_season",
    "validate_file_extension",
    "validate_inputs",
]
