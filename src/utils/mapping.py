# ============================================================
# File: src/utils/mapping.py
# Purpose: Map TEAM_ID, PLAYER_ID to human-readable names
# Version: 1.0
# ============================================================

import pandas as pd

TEAM_MAP = {i: f"Team_{i}" for i in range(30)}
PLAYER_MAP = {i: f"Player_{i}" for i in range(1000)}

def map_team_ids(df, team_col="TEAM_ID"):
    if team_col in df.columns:
        df["TEAM_NAME"] = df[team_col].map(TEAM_MAP).fillna("UnknownTeam")
    return df

def map_player_ids(df, player_col="PLAYER_ID"):
    if player_col in df.columns:
        df["PLAYER_NAME"] = df[player_col].map(PLAYER_MAP).fillna("UnknownPlayer")
    return df
