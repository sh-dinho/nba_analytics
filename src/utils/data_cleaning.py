# ============================================================
# File: src/utils/data_cleaning.py
# Purpose: Utility functions for cleaning and renaming NBA game data
# Project: nba_analysis
# Version: 1.1 (adds dependencies section + defensive handling)
#
# Dependencies:
# - pandas
# ============================================================

import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean NBA game data by converting GAME_DATE to datetime and dropping rows with NaN values.

    Arguments:
    df -- DataFrame containing NBA game data

    Returns:
    DataFrame -- Cleaned DataFrame
    """
    df = df.copy()

    # Convert GAME_DATE to datetime safely
    if "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")

    # Drop rows with NaN values
    df = df.dropna()

    return df


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns to standardized names for consistency.

    Arguments:
    df -- DataFrame containing NBA game data

    Returns:
    DataFrame -- DataFrame with renamed columns
    """
    df = df.copy()

    rename_map = {
        "PTS": "POINTS",
        "WL": "TARGET",
    }

    # Only rename if columns exist
    existing_map = {k: v for k, v in rename_map.items() if k in df.columns}
    if existing_map:
        df.rename(columns=existing_map, inplace=True)

    return df
