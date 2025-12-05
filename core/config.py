# ============================================================
# File: core/config.py
# Purpose: Global configuration constants for NBA analytics pipeline
# ============================================================

from core.paths import (
    MODEL_FILE_PKL,
    XGB_ML_MODEL_FILE,
    XGB_OU_MODEL_FILE,
    ENSEMBLE_MODEL_FILE,
    PLAYER_MODEL_FILE,
    TEAM_MODEL_FILE,
    SUMMARY_FILE,
    PICKS_FILE,
    PICKS_BANKROLL_FILE,
    FEATURES_FILE,       
    FEATURES_LOG_FILE,
    CONFIG_LOG_FILE,
    RESULTS_DIR,
)

# === Simulation defaults ===
DEFAULT_BANKROLL = 1000.0
MAX_KELLY_FRACTION = 0.05
EV_THRESHOLD = 0.0
MIN_KELLY_STAKE = 1.0

# === Output behavior ===
PRINT_ONLY_ACTIONABLE = False