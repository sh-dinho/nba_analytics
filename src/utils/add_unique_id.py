# ============================================================
# File: src/utils/add_unique_id.py
# Purpose: Add unique identifier column to features DataFrame
# Project: nba_analysis
# Version: 1.4 (deduplication + type enforcement)
# ============================================================

import logging
import datetime
import pandas as pd


def add_unique_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a unique_id column combining GAME_ID, TEAM_ID, and prediction_date.
    Guarantees unique_id even if GAME_ID/TEAM_ID are missing.
    Deduplicates rows with identical unique_id values.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("add_unique_id expects a pandas DataFrame")

    df = df.copy()

    if "GAME_ID" not in df.columns:
        logging.warning("GAME_ID missing, assigning placeholder")
        df["GAME_ID"] = [f"unknown_game_{i}" for i in range(len(df))]
    df["GAME_ID"] = df["GAME_ID"].astype(str)

    if "TEAM_ID" not in df.columns:
        logging.warning("TEAM_ID missing, assigning placeholder")
        df["TEAM_ID"] = -1
    df["TEAM_ID"] = pd.to_numeric(df["TEAM_ID"], errors="coerce").fillna(-1).astype(int)

    if "prediction_date" not in df.columns:
        logging.warning("prediction_date missing, assigning today's date")
        df["prediction_date"] = datetime.date.today().isoformat()
    df["prediction_date"] = df["prediction_date"].astype(str)

    df["unique_id"] = (
        df["GAME_ID"].astype(str)
        .str.cat(df["TEAM_ID"].astype(str), sep="_")
        .str.cat(df["prediction_date"].astype(str), sep="_")
    )

    # Deduplicate on unique_id
    before = len(df)
    df = df.drop_duplicates(subset=["unique_id"]).reset_index(drop=True)
    after = len(df)

    logging.info("Generated %d unique IDs (deduplicated from %d rows). Example: %s",
                 after, before, df["unique_id"].iloc[0] if not df.empty else "N/A")
    return df
