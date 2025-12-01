# config.py
import os
import json
import logging
from typing import Any, Dict

# Environment-overridable paths and keys
DB_PATH = os.getenv("DB_PATH", "data/nba.sqlite")
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "YOUR_PLACEHOLDER_KEY")
TEAM_ALIAS_PATH = os.getenv("TEAM_ALIAS_PATH", "config/team_aliases.json")

# Prediction & strategy constants (env overrides)
THRESHOLD = float(os.getenv("PREDICTION_THRESHOLD", 0.6))
MAX_KELLY_FRACTION = float(os.getenv("MAX_KELLY_FRACTION", 0.05))

# Logging defaults (can be overridden by config.yaml)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = os.getenv("LOG_FORMAT", "%(asctime)s [%(levelname)s] %(message)s")

def configure_logging(level: str = LOG_LEVEL, fmt: str = LOG_FORMAT) -> None:
    logging.basicConfig(level=getattr(logging, level), format=fmt)

def load_team_aliases() -> Dict[str, str]:
    try:
        with open(TEAM_ALIAS_PATH, "r") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("Team alias file must contain a JSON object")
            return data
    except FileNotFoundError:
        logging.warning(f"Team alias file not found at {TEAM_ALIAS_PATH}. Using empty map.")
        return {}
    except ValueError as e:
        logging.error(f"Invalid team alias file: {e}")
        return {}

TEAM_MAP = load_team_aliases()