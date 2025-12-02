# core/config.py
import os

# ----------------------------
# Paths
# ----------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../data")
MODELS_DIR = os.path.join(BASE_DIR, "../models")
RESULTS_DIR = os.path.join(BASE_DIR, "../results")

# ----------------------------
# General settings
# ----------------------------
SEED = 42
MAX_KELLY_FRACTION = 0.05
STRONG_PICK_THRESHOLD = 0.6

# ----------------------------
# External APIs / Sources
# ----------------------------
PLAYER_STATS_URL = "https://api.example.com/nba/player_stats"
ODDS_SOURCE_URL = "https://api.example.com/nba/odds"
