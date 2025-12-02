# File: core/config.py
# Centralized configuration constants for the NBA analytics pipeline.

import os

# -----------------------------
# Base Directories & Environment
# -----------------------------
# Detect environment (default: local)
ENV = os.getenv("PIPELINE_ENV", "local")

# Define paths relative to the project root (assuming this file is in a 'core' folder)
PROJECT_ROOT = os.path.dirname(os.path.abspath(os.path.join(__file__, '..')))

if ENV == "ci":
    # CI/CD environment paths - relative to the root of the repository
    BASE_DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    BASE_MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
    BASE_RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
    BASE_LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
    DB_PATH = os.path.join(PROJECT_ROOT, "Data/TeamData.sqlite") # Path for CI environment
else: 
    # Local dev defaults (using user home directory for persistence)
    BASE_DIR = os.path.expanduser("~/nba_analytics")
    BASE_DATA_DIR = os.path.join(BASE_DIR, "data")
    BASE_MODELS_DIR = os.path.join(BASE_DIR, "models")
    BASE_RESULTS_DIR = os.path.join(BASE_DIR, "results")
    BASE_LOGS_DIR = os.path.join(BASE_DIR, "logs")
    # CRITICAL FIX: The DB path is also centralized here.
    DB_PATH = os.path.join(BASE_DIR, "Data/TeamData.sqlite") 

def ensure_dirs():
    """Ensure all core directories are created."""
    for d in [BASE_DATA_DIR, BASE_MODELS_DIR, BASE_RESULTS_DIR, BASE_LOGS_DIR]:
        os.makedirs(d, exist_ok=True)
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# -----------------------------
# Model artifacts
# -----------------------------
MODEL_FILE_PKL = os.path.join(BASE_MODELS_DIR, "game_predictor.pkl")
MODEL_FILE_H5 = os.path.join(BASE_MODELS_DIR, "game_predictor.h5") 

# -----------------------------
# Data files
# -----------------------------
NEW_GAMES_FILE = os.path.join(BASE_DATA_DIR, "new_games.csv")
PLAYER_STATS_FILE = os.path.join(BASE_DATA_DIR, "player_stats.csv")
TRAINING_FEATURES_FILE = os.path.join(BASE_DATA_DIR, "training_features.csv")
GAME_RESULTS_FILE = os.path.join(BASE_DATA_DIR, "game_results.csv") 

# -----------------------------
# Results files
# -----------------------------
PREDICTIONS_FILE = os.path.join(BASE_RESULTS_DIR, "today_predictions.csv")
DAILY_SUMMARY_FILE = os.path.join(BASE_RESULTS_DIR, "pipeline_summary.csv")
BANKROLL_FILE_TEMPLATE = os.path.join(BASE_RESULTS_DIR, "picks_bankroll_{model_type}.csv")
MONTHLY_SUMMARY_FILE = os.path.join(BASE_RESULTS_DIR, "monthly_summary.csv")

# ----------------------------
# General settings
# ----------------------------
SEED = 42
MAX_KELLY_FRACTION = 0.05
STRONG_PICK_THRESHOLD = 0.6