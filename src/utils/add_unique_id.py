# ============================================================
# File: src/utils/add_unique_id.py
# Purpose: Add unique identifier column to features DataFrame
# Project: nba_analysis
# Version: 1.2 (adds dependencies section + defensive handling)
#
# Dependencies:
# - logging (standard library)
# - datetime (standard library)
# - pandas (recommended for DataFrame operations)
# ============================================================

import logging
import datetime


def add_unique_id(df):
    """
    Add a unique_id column combining GAME_ID, TEAM_ID, and prediction_date.
    Guarantees unique_id even if GAME_ID/TEAM_ID are missing.

    Arguments:
    df -- DataFrame containing the features of the game, including GAME_ID, TEAM_ID, and prediction_date

    Returns:
    DataFrame -- The input DataFrame with an added unique_id column
    """

    # Defensive copy to avoid mutating the original DataFrame
    df = df.copy()

    # Ensure GAME_ID exists, create placeholder if missing
    if "GAME_ID" not in df.columns:
        logging.warning("GAME_ID missing, assigning placeholder")
        df["GAME_ID"] = [f"unknown_game_{i}" for i in range(len(df))]

    # Ensure TEAM_ID exists, create placeholder if missing
    if "TEAM_ID" not in df.columns:
        logging.warning("TEAM_ID missing, assigning placeholder")
        df["TEAM_ID"] = -1

    # Ensure prediction_date exists, assign today's date if missing
    if "prediction_date" not in df.columns:
        logging.warning("prediction_date missing, assigning today's date")
        df["prediction_date"] = datetime.date.today().isoformat()

    # Create a unique_id by combining GAME_ID, TEAM_ID, and prediction_date
    df["unique_id"] = (
        df["GAME_ID"].astype(str)
        + "_"
        + df["TEAM_ID"].astype(str)
        + "_"
        + df["prediction_date"].astype(str)
    )

    logging.info(f"Unique IDs generated for {len(df)} rows.")
    return df
