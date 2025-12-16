# ============================================================
# File: src/schedule/contract.py
# Purpose: Canonical TEAM-level schedule schema & validation
# Project: nba_analysis
# ============================================================

import pandas as pd

# ------------------------------------------------------------------
# Canonical TEAM-level schedule columns
# ------------------------------------------------------------------
TEAM_SCHEDULE_COLUMNS = [
    "GAME_ID",
    "TEAM_ID",
    "GAME_DATE",
    "HOME_TEAM",
    "AWAY_TEAM",
    "PTS",
    "WL",
]


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------
def validate_team_schedule(df: pd.DataFrame, enforce_types=True) -> None:
    """Validate canonical TEAM-level schedule."""
    required = set(TEAM_SCHEDULE_COLUMNS)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required TEAM-level columns: {missing}")

    if df.duplicated(subset=["GAME_ID", "TEAM_ID"]).any():
        raise ValueError("Duplicate GAME_ID × TEAM_ID rows detected.")

    if df["GAME_DATE"].isna().any():
        raise ValueError("Null GAME_DATE values are not allowed.")

    if "WL" in df.columns:
        invalid = ~df["WL"].isin(["W", "L"])
        if invalid.any():
            raise ValueError("Invalid WL values found (expected 'W' or 'L').")

    if enforce_types:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="raise")
        df["TEAM_ID"] = df["TEAM_ID"].astype(int)
        df["HOME_TEAM"] = df["HOME_TEAM"].astype(int)
        df["AWAY_TEAM"] = df["AWAY_TEAM"].astype(int)
        df["PTS"] = pd.to_numeric(df["PTS"], errors="raise")


def sort_team_schedule(df: pd.DataFrame) -> pd.DataFrame:
    """Sort schedule by TEAM_ID, GAME_DATE ascending."""
    return df.sort_values(["TEAM_ID", "GAME_DATE"]).reset_index(drop=True)


def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return a small summary: games, teams, avg points, max win streak placeholder."""
    stats = {
        "total_games": df["GAME_ID"].nunique(),
        "total_teams": df["TEAM_ID"].nunique(),
        "avg_pts": df["PTS"].mean(),
    }
    return pd.DataFrame([stats])
