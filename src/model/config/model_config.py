from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Model Configuration
# File: src/model/config/model_config.py
# Author: Sadiq
#
# Description:
#     Central configuration for model training, prediction,
#     and model registry metadata.
# ============================================================

# ------------------------------------------------------------
# Model identity
# ------------------------------------------------------------
MODEL_NAME = "nba_winprob_model"
DEFAULT_MODEL_VERSION = 1

# ------------------------------------------------------------
# Target + Features
# ------------------------------------------------------------
TARGET = "won"

FEATURES = [
    "elo",
    "opp_elo",
    "score_diff",
    "roll_margin_5",
    "roll_margin_10",
    "roll_win_rate_5",
    "roll_win_rate_10",
    "rest_days",
    "is_home",
    "form_last3",
    "sos",
]

# ------------------------------------------------------------
# Training parameters
# ------------------------------------------------------------
TRAINING_CONFIG = {
    "test_size": 0.20,
    "random_state": 42,
    "calibration_bins": 10,
}

# ------------------------------------------------------------
# Supported model types
# ------------------------------------------------------------
SUPPORTED_MODELS = [
    "xgboost",
    "lightgbm",
    "logistic_regression",
]