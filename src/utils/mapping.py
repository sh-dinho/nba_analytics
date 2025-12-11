# ============================================================
# File: src/utils/mapping.py
# Purpose: Map TEAM_ID, PLAYER_ID to human-readable names
# Project: nba_analysis
# Version: 1.3 (adds combined helper map_ids)
# ============================================================

import pandas as pd
import logging

# Default synthetic mapping dictionaries
TEAM_MAP = {i: f"Team_{i}" for i in range(30)}
PLAYER_MAP = {i: f"Player_{i}" for i in range(1000)}


def map_team_ids(
    df: pd.DataFrame, team_col: str = "TEAM_ID", team_map: dict = None
) -> pd.DataFrame:
    """Map TEAM_ID values in a DataFrame to human-readable team names."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("map_team_ids expects a pandas DataFrame")

    df = df.copy()
    team_map = team_map or TEAM_MAP

    if team_col in df.columns:
        df["TEAM_NAME"] = df[team_col].map(team_map).fillna("UnknownTeam")
    else:
        df["TEAM_NAME"] = "UnknownTeam"

    mapped_count = (df["TEAM_NAME"] != "UnknownTeam").sum()
    logging.info("Mapped %d team IDs, %d unknown", mapped_count, len(df) - mapped_count)

    return df


def map_player_ids(
    df: pd.DataFrame, player_col: str = "PLAYER_ID", player_map: dict = None
) -> pd.DataFrame:
    """Map PLAYER_ID values in a DataFrame to human-readable player names."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("map_player_ids expects a pandas DataFrame")

    df = df.copy()
    player_map = player_map or PLAYER_MAP

    if player_col in df.columns:
        df["PLAYER_NAME"] = df[player_col].map(player_map).fillna("UnknownPlayer")
    else:
        df["PLAYER_NAME"] = "UnknownPlayer"

    mapped_count = (df["PLAYER_NAME"] != "UnknownPlayer").sum()
    logging.info(
        "Mapped %d player IDs, %d unknown", mapped_count, len(df) - mapped_count
    )

    return df


def map_ids(
    df: pd.DataFrame,
    team_col: str = "TEAM_ID",
    player_col: str = "PLAYER_ID",
    team_map: dict = None,
    player_map: dict = None,
) -> pd.DataFrame:
    """
    Combined helper: map both TEAM_ID and PLAYER_ID to human-readable names in one call.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("map_ids expects a pandas DataFrame")

    df = map_team_ids(df, team_col=team_col, team_map=team_map)
    df = map_player_ids(df, player_col=player_col, player_map=player_map)
    logging.info("Mapped both team and player IDs for %d rows", len(df))
    return df
