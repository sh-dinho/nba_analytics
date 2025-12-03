# ============================================================
# File: core/config.py
# Purpose: Centralized configuration for NBA analytics pipeline
# ============================================================

from pathlib import Path
import os
import logging

ENV = os.getenv("PIPELINE_ENV", "local").lower()
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if ENV in ("ci", "prod"):
    BASE_DATA_DIR = PROJECT_ROOT / "data"
    BASE_MODELS_DIR = PROJECT_ROOT / "models"
    BASE_RESULTS_DIR = PROJECT_ROOT / "results"
    BASE_LOGS_DIR = PROJECT_ROOT / "logs"
    DB_PATH = PROJECT_ROOT / "Data" / "TeamData.sqlite"
else:
    BASE_DIR = Path.home() / "nba_analytics"
    BASE_DATA_DIR = BASE_DIR / "data"
    BASE_MODELS_DIR = BASE_DIR / "models"
    BASE_RESULTS_DIR = BASE_DIR / "results"
    BASE_LOGS_DIR = BASE_DIR / "logs"
    DB_PATH = BASE_DIR / "Data" / "TeamData.sqlite"

# Backward compatibility aliases
RESULTS_DIR = BASE_RESULTS_DIR
MODELS_DIR = BASE_MODELS_DIR

def ensure_dirs():
    for d in [BASE_DATA_DIR, BASE_MODELS_DIR, BASE_RESULTS_DIR, BASE_LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

def dump_config():
    return {
        "ENV": ENV,
        "BASE_DATA_DIR": str(BASE_DATA_DIR),
        "BASE_MODELS_DIR": str(BASE_MODELS_DIR),
        "BASE_RESULTS_DIR": str(BASE_RESULTS_DIR),
        "BASE_LOGS_DIR": str(BASE_LOGS_DIR),
        "DB_PATH": str(DB_PATH),
        "SEED": SEED,
        "DEFAULT_THRESHOLD": DEFAULT_THRESHOLD,
        "DEFAULT_BANKROLL": DEFAULT_BANKROLL,
        "MAX_KELLY_FRACTION": MAX_KELLY_FRACTION,
    }

# Logging setup
logging.basicConfig(
    filename=BASE_LOGS_DIR / "pipeline.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Model artifacts
MODEL_FILE_PKL = BASE_MODELS_DIR / "game_predictor.pkl"
MODEL_FILE_H5 = BASE_MODELS_DIR / "game_predictor.h5"

# Data files
TRAINING_FEATURES_FILE = BASE_DATA_DIR / "training_features.csv"
NEW_GAMES_FILE = BASE_DATA_DIR / "new_games.csv"
NEW_GAMES_FEATURES_FILE = BASE_DATA_DIR / "new_games_features.csv"
HISTORICAL_GAMES_FILE = BASE_DATA_DIR / "historical_games.csv"
PLAYER_STATS_FILE = BASE_DATA_DIR / "player_stats.csv"
GAME_RESULTS_FILE = BASE_DATA_DIR / "game_results.csv"

# Results files
PREDICTIONS_FILE = BASE_RESULTS_DIR / "today_predictions.csv"
BANKROLL_FILE_TEMPLATE = BASE_RESULTS_DIR / "picks_bankroll_{model_type}.csv"
PICKS_FILE = BASE_RESULTS_DIR / "picks.csv"
PICKS_LOG = BASE_LOGS_DIR / "picks.log"
SUMMARY_FILE = BASE_RESULTS_DIR / "summary.csv"

# General settings
SEED = 42
DEFAULT_THRESHOLD = 0.6
DEFAULT_BANKROLL = 1000.0
MAX_KELLY_FRACTION = 0.05