# ============================================================
# Path: src/utils/add_unique_id.py
# Purpose: Add unique identifier column to features DataFrame
# Version: 1.1 (patched to auto-fix missing IDs)
# ============================================================

import logging

def add_unique_id(df):
    """
    Add a unique_id column combining GAME_ID, TEAM_ID, and prediction_date.
    Guarantees unique_id even if GAME_ID/TEAM_ID are missing.
    """
    if "GAME_ID" not in df.columns:
        logging.warning("GAME_ID missing, assigning placeholder")
        df["GAME_ID"] = [f"unknown_game_{i}" for i in range(len(df))]

    if "TEAM_ID" not in df.columns:
        logging.warning("TEAM_ID missing, assigning placeholder")
        df["TEAM_ID"] = -1

    if "prediction_date" not in df.columns:
        logging.warning("prediction_date missing, assigning today's date")
        import datetime
        df["prediction_date"] = datetime.date.today().isoformat()

    df["unique_id"] = (
        df["GAME_ID"].astype(str)
        + "_"
        + df["TEAM_ID"].astype(str)
        + "_"
        + df["prediction_date"].astype(str)
    )
    return df
