from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Model Configuration
# File: src/model/config/model_config.py
# Author: Sadiq
# ============================================================

# ------------------------------------------------------------
# Config identity
# ------------------------------------------------------------
MODEL_CONFIG_VERSION = "2026.01"
MODEL_NAME = "nba_winprob_model"

DEFAULT_MODEL_VERSION = 1
DEFAULT_FEATURE_VERSION = "v5"
DEFAULT_MODEL_FAMILY = "xgboost"

# ------------------------------------------------------------
# Target columns for each model type
# (aligned with FEATURES_SNAPSHOT)
# ------------------------------------------------------------
TARGET_MAP = {
    "moneyline": "win",          # was "won" — corrected
    "totals": "total_points",
    "spread": "margin",
}

MODEL_TYPES = list(TARGET_MAP.keys())

# ------------------------------------------------------------
# Feature sets for each model type
# (aligned EXACTLY with FEATURES_SNAPSHOT)
# ------------------------------------------------------------
BASE_FEATURES = [
    # Elo + rolling Elo
    "elo",
    "elo_roll5",
    "elo_roll10",
    "opp_elo",

    # Margin + rolling margin
    "margin",
    "margin_rolling_5",
    "margin_rolling_10",
    "margin_rolling_20",

    # Win-rate rolling
    "win_rolling_5",
    "win_rolling_20",

    # Points for (rolling)
    "points_for_rolling_5",
    "points_for_rolling_10",
    "points_for_rolling_20",

    # Points against (rolling)
    "points_against_rolling_5",
    "points_against_rolling_10",
    "points_against_rolling_20",

    # Opponent rolling features
    "opp_margin_rolling_5",
    "opp_margin_rolling_10",
    "opp_win_pct_last10",

    # Team form
    "team_win_pct_last10",
    "win_streak",
    "form_last3",

    # Contextual
    "rest_days",
    "is_home",
    "is_b2b",
    "sos",
]

FEATURE_MAP = {
    "moneyline": BASE_FEATURES,
    "totals": BASE_FEATURES,
    "spread": BASE_FEATURES,
}

# Optional convenience alias
DEFAULT_FEATURES = FEATURE_MAP["moneyline"]

# ------------------------------------------------------------
# Training parameters
# ------------------------------------------------------------
TRAINING_CONFIG = {
    "test_size": 0.20,
    "random_state": 42,
    "calibration_bins": 10,
}

# ------------------------------------------------------------
# Supported model families
# ------------------------------------------------------------
SUPPORTED_MODEL_FAMILIES = [
    "xgboost",
    "lightgbm",
    "logistic_regression",
]

# ------------------------------------------------------------
# Exported constants
# ------------------------------------------------------------
SUPPORTED_MODEL_TYPES = MODEL_TYPES