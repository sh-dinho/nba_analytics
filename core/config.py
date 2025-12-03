# ============================================================
# File: core/config.py
# Purpose: Centralized configuration for NBA analytics pipeline
# ============================================================

from pathlib import Path
import os

ENV = os.getenv("PIPELINE_ENV", "local")
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if ENV == "ci":
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

# Model artifacts
MODEL_FILE_PKL = BASE_MODELS_DIR / "game_predictor.pkl"
MODEL_FILE_H5 = BASE_MODELS_DIR / "game_predictor.h5"

# Data files
TRAINING_FEATURES_FILE = BASE_DATA_DIR / "training_features.csv"
PLAYER_STATS_FILE = BASE_DATA_DIR / "player_stats.csv"
GAME_RESULTS_FILE = BASE_DATA_DIR / "game_results.csv"

# ✅ Added constant for new games file
NEW_GAMES_FILE = BASE_DATA_DIR / "new_games.csv"

# Results files
PREDICTIONS_FILE = BASE_RESULTS_DIR / "today_predictions.csv"
BANKROLL_FILE_TEMPLATE = BASE_RESULTS_DIR / "picks_bankroll_{model_type}.csv"
TODAY_PREDICTIONS_FILE = BASE_RESULTS_DIR / "predictions.csv"
PICKS_FILE = BASE_RESULTS_DIR / "picks.csv"
PICKS_LOG = BASE_LOGS_DIR / "picks.log"

# General settings
SEED = 42
DEFAULT_THRESHOLD = 0.6
DEFAULT_BANKROLL = 1000.0
MAX_KELLY_FRACTION = 0.05