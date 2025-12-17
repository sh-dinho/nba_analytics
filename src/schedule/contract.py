# ============================================================
# File: src/schedule/contract.py
# Purpose: Validation contract for team schedule schema (v2.0)
# Version: 2.0
# Author: Your Team
# Date: December 2025
# ============================================================

import pandas as pd


def validate_team_schedule(df: pd.DataFrame):
    required = [
        "gameId",
        "seasonYear",
        "startDate",
        "homeTeam",
        "awayTeam",
        "homeScore",
        "awayScore",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Schedule validation failed, missing: {missing}")
    if df.empty:
        raise ValueError("Schedule validation failed: dataframe is empty.")
    if df["startDate"].isna().any():
        raise ValueError("Schedule validation failed: startDate contains NA.")
