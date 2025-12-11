# ============================================================
# File: src/utils/mapping.py
# Purpose: Map TEAM_ID, PLAYER_ID to human-readable names
# Project: nba_analysis
# Version: 1.1 (adds dependencies section + defensive handling)
#
# Dependencies:
# - pandas
# ============================================================

import pandas as pd

# Mapping dictionaries for teams and players
TEAM_MAP = {i: f"Team_{i}" for i in range(30)}   # Maps TEAM_ID to human-readable team names
PLAYER_MAP = {i: f"Player_{i}" for i in range(1000)}  # Maps PLAYER_ID to human-readable player names


def map_team_ids(df: pd.DataFrame, team_col: str = "TEAM_ID") -> pd.DataFrame:
    """
    Maps TEAM_ID values in a DataFrame to human-readable team names.

    Args:
        df (pd.DataFrame): Input DataFrame with a TEAM_ID column.
        team_col (str): Name of the column containing TEAM_ID values (default is "TEAM_ID").

    Returns:
        pd.DataFrame: DataFrame with an additional column 'TEAM_NAME' mapped to team names.
    """
    df = df.copy()
    if team_col in df.columns:
        df["TEAM_NAME"] = df[team_col].map(TEAM_MAP).fillna("UnknownTeam")
    else:
        df["TEAM_NAME"] = "UnknownTeam"
    return df


def map_player_ids(df: pd.DataFrame, player_col: str = "PLAYER_ID") -> pd.DataFrame:
    """
    Maps PLAYER_ID values in a DataFrame to human-readable player names.

    Args:
        df (pd.DataFrame): Input DataFrame with a PLAYER_ID column.
        player_col (str): Name of the column containing PLAYER_ID values (default is "PLAYER_ID").

    Returns:
        pd.DataFrame: DataFrame with an additional column 'PLAYER_NAME' mapped to player names.
    """
    df = df.copy()
    if player_col in df.columns:
        df["PLAYER_NAME"] = df[player_col].map(PLAYER_MAP).fillna("UnknownPlayer")
    else:
        df["PLAYER_NAME"] = "UnknownPlayer"
    return df
