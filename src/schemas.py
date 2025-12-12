# ============================================================
# File: src/schemas.py
# Purpose: Centralized schema definitions + helpers
# ============================================================

import pandas as pd

# Historical schedule schema (multi-season fetch)
HISTORICAL_SCHEDULE_COLUMNS = [
    "GAME_ID",
    "GAME_DATE",
    "TEAM_ID",
    "TEAM_NAME",
    "MATCHUP",
    "PTS",
    "PTS_OPP",
    "WL",  # Original win/loss string
    "SEASON",  # Season string or int
]

# Enriched schedule schema (single season, home/visitor split)
ENRICHED_SCHEDULE_COLUMNS = [
    "GAME_ID",
    "GAME_DATE",
    "HOME_TEAM_ID",
    "VISITOR_TEAM_ID",
    "PTS",
    "PTS_OPP",
    "WL",  # Original win/loss string
    "WIN",  # Numeric win flag
    "SEASON",
]

# Today's schedule schema (short-term games)
TODAY_SCHEDULE_COLUMNS = [
    "GAME_ID",
    "GAME_DATE_EST",
    "HOME_TEAM_ABBREVIATION",
    "AWAY_TEAM_ABBREVIATION",
    "GAME_TYPE",
]

# Feature schema (for model training)
FEATURE_COLUMNS = [
    "GAME_ID",
    "SEASON",
    "PTS",
    "PTS_OPP",
    "WL",  # Original string
    "WIN",  # Numeric target
    "AVG_PTS_LAST3",
    "AVG_PTS_ALLOWED_LAST3",
    "WIN_STREAK",
]


def normalize_df(df: pd.DataFrame, expected_cols: list[str]) -> pd.DataFrame:
    """Ensure DataFrame has all expected columns, fill missing with None."""
    for col in expected_cols:
        if col not in df.columns:
            df[col] = None
    return df[expected_cols]


def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure features DataFrame has all expected columns."""
    return normalize_df(df, FEATURE_COLUMNS)


def normalize_today_schedule(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure today's schedule DataFrame has all expected columns."""
    return normalize_df(df, TODAY_SCHEDULE_COLUMNS)
