# ============================================================
# File: scripts/query_teamdata_all.py
# Purpose: Query the unified teamdata_all table for multi-season feature engineering
# ============================================================

import os
import sqlite3
import pandas as pd
from core.config import DB_PATH
from core.log_config import setup_logger
from core.exceptions import PipelineError, DataError
from core.utils import ensure_columns

logger = setup_logger("query_teamdata_all")


def query_teamdata_all() -> pd.DataFrame:
    """Query the unified teamdata_all table and perform feature engineering across seasons."""
    if not os.path.exists(DB_PATH):
        raise PipelineError(f"Database not found at {DB_PATH}")

    try:
        con = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM teamdata_all", con)
    except Exception as e:
        raise PipelineError(f"Failed to query teamdata_all: {e}")
    finally:
        con.close()

    # Validate required columns
    try:
        ensure_columns(df, {"Season", "Date", "TEAM_ID", "PTS", "AST", "REB", "Days-Rest-Home"}, "teamdata_all")
    except ValueError as e:
        raise DataError(str(e))

    # Example 1: Filter by season
    df_2022 = df[df["Season"] == "2022-23"]
    logger.info(f"Filtered 2022-23 season: {len(df_2022)} rows")

    # Example 2: Compute rolling averages across seasons
    df_sorted = df.sort_values(["Season", "Date", "TEAM_ID"])
    df_sorted["PTS_rolling_10"] = (
        df_sorted.groupby("TEAM_ID")["PTS"]
                 .transform(lambda x: x.rolling(10, min_periods=1).mean())
    )

    # Example 3: Rest-day impact across seasons
    df_sorted["RestImpact"] = df_sorted["Days-Rest-Home"] * df_sorted["PTS"]

    # Example 4: Aggregate stats by season
    season_summary = (
        df_sorted.groupby("Season")[["PTS", "AST", "REB"]]
                 .mean()
                 .reset_index()
    )

    logger.info("Season summary (avg stats):")
    logger.info(f"\n{season_summary}")

    return df_sorted


if __name__ == "__main__":
    query_teamdata_all()