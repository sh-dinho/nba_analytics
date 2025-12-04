# ============================================================
# File: core/config.py
# Purpose: Pipeline constants and configuration
# ============================================================

from core.paths import (
    MODEL_FILE_PKL,
    FEATURES_FILE,
    PREDICTIONS_FILE,
    PICKS_FILE,
    PICKS_BANKROLL_FILE,
    SUMMARY_FILE,
    TRAINING_FEATURES_FILE,
    PLAYER_FEATURES_FILE,
    NEW_GAMES_FEATURES_FILE,
    FEATURES_LOG_FILE,
    LOGS_DIR,
    XGB_ML_MODEL_FILE,
    XGB_OU_MODEL_FILE,
    ENSEMBLE_MODEL_FILE,
    PLAYER_MODEL_FILE,
    TEAM_MODEL_FILE,
)

import pandas as pd
from datetime import datetime

# -----------------------------
# Pipeline settings
# -----------------------------
DEFAULT_BANKROLL = 1000.0      # Starting bankroll
MAX_KELLY_FRACTION = 0.1       # Max fraction of bankroll per pick
PREDICTION_THRESHOLD = 0.6     # Probability threshold for picks

# -----------------------------
# Feature building
# -----------------------------
USE_ROLLING_AVG = True         # Toggle rolling averages vs. season averages
ROLLING_WINDOW = 5             # Rolling averages window for stats

# -----------------------------
# Model target settings
# -----------------------------
TARGET_LABEL = "label"          # Classification
TARGET_MARGIN = "margin"        # Regression
TARGET_CATEGORY = "outcome_category"  # Multi-class

# -----------------------------
# Optional
# -----------------------------
SUMMARY_COLUMNS = [
    "Date",
    "Total_Stake",
    "Avg_EV",
    "Bankroll_Change"
]

# -----------------------------
# Config logging
# -----------------------------
CONFIG_LOG_FILE = LOGS_DIR / "config_summary.csv"


# -----------------------------
# Feature logs
# -----------------------------
FEATURES_LOG_FILE = LOGS_DIR / "features_summary.csv"

def log_config_snapshot():
    """Append current configuration values to CONFIG_LOG_FILE."""
    snapshot = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "USE_ROLLING_AVG": USE_ROLLING_AVG,
        "ROLLING_WINDOW": ROLLING_WINDOW,
        "PREDICTION_THRESHOLD": PREDICTION_THRESHOLD,
        "MAX_KELLY_FRACTION": MAX_KELLY_FRACTION,
        "DEFAULT_BANKROLL": DEFAULT_BANKROLL,
    }
    df = pd.DataFrame([snapshot])
    if CONFIG_LOG_FILE.exists():
        df.to_csv(CONFIG_LOG_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(CONFIG_LOG_FILE, index=False)