# ============================================================
# File: src/utils/data_cleaning.py
# Purpose: Utility functions for cleaning and renaming NBA game data
# Project: nba_analysis
# Version: 1.3 (adds combined helper + consistent headers)
#
# Dependencies:
# - pandas
# - logging (for info/debug messages)
# ============================================================

import pandas as pd
import logging


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean NBA game data by converting GAME_DATE to datetime and dropping rows
    missing critical identifiers (GAME_DATE, TEAM_ID, GAME_ID).
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("clean_data expects a pandas DataFrame")

    df = df.copy()

    if "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")

    critical_cols = [c for c in ["GAME_DATE", "TEAM_ID", "GAME_ID"] if c in df.columns]
    before = len(df)
    if critical_cols:
        df = df.dropna(subset=critical_cols)
    after = len(df)
    logging.info("Dropped %d rows with missing critical values", before - after)

    return df


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns to standardized names for consistency.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("rename_columns expects a pandas DataFrame")

    df = df.copy()
    df.columns = [c.upper() for c in df.columns]

    rename_map = {
        "PTS": "POINTS",
        "WL": "TARGET",
        "REB": "REBOUNDS",
        "AST": "ASSISTS",
    }

    existing_map = {k: v for k, v in rename_map.items() if k in df.columns}
    if existing_map:
        df.rename(columns=existing_map, inplace=True)
        logging.info("Renamed columns: %s", existing_map)

    return df


def prepare_game_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combined helper: always clean and rename NBA game data in one call.
    """
    df = clean_data(df)
    df = rename_columns(df)
    logging.info("Game data prepared with %d rows and standardized columns", len(df))
    return df
