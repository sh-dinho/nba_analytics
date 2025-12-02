# File: config.py
# Centralized configuration constants for the NBA analytics pipeline.
# Place this file at the root of the repository (same level as scripts/).

import os

# Detect environment (default: local)
ENV = os.getenv("PIPELINE_ENV", "local")

# -----------------------------
# Base directories
# -----------------------------
if ENV == "ci":
    BASE_DATA_DIR = "data/"
    BASE_MODELS_DIR = "models/"
    BASE_RESULTS_DIR = "results/"
    BASE_LOGS_DIR = "logs/"
else:  # local dev defaults
    BASE_DATA_DIR = os.path.expanduser("~/nba_analytics/data/")
    BASE_MODELS_DIR = os.path.expanduser("~/nba_analytics/models/")
    BASE_RESULTS_DIR = os.path.expanduser("~/nba_analytics/results/")
    BASE_LOGS_DIR = os.path.expanduser("~/nba_analytics/logs/")

# -----------------------------
# Model artifact
# -----------------------------
MODEL_FILE = os.path.join(BASE_MODELS_DIR, "game_predictor.pkl")

# -----------------------------
# Data files
# -----------------------------
PLAYER_STATS_FILE = os.path.join(BASE_DATA_DIR, "player_stats.csv")
TRAINING_FEATURES_FILE = os.path.join(BASE_DATA_DIR, "training_features.csv")
GAME_RESULTS_FILE = os.path.join(BASE_DATA_DIR, "game_results.csv")

# -----------------------------
# Results files
# -----------------------------
PREDICTIONS_FILE = os.path.join(BASE_RESULTS_DIR, "predictions.csv")
PICKS_FILE = os.path.join(BASE_RESULTS_DIR, "picks.csv")
PICKS_BANKROLL_FILE = os.path.join(BASE_RESULTS_DIR, "picks_bankroll.csv")

# -----------------------------
# Logs directory
# -----------------------------
LOGS_DIR = BASE_LOGS_DIR