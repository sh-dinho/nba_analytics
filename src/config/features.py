# ============================================================
# Project: NBA Analytics & Betting Engine
# Author: Sadiq
# Description: Feature definitions and versioning for model input.
# ============================================================

FEATURE_VERSION = "v1"

BASE_FEATURES = [
    "is_home",
    "rolling_win_rate",
    "rolling_points_for",
    "rolling_points_against",
]

ROLLING_WINDOWS = {
    "win_rate": 10,
    "points": 10,
}

REQUIRED_COLUMNS_LONG = [
    "game_id",
    "date",
    "team",
    "opponent",
    "is_home",
    "points_for",
    "points_against",
    "won",
    "season",
    "game_number",
]
