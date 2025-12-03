# ============================================================
# File: scripts/merge_team_data.py
# Purpose: Merge all season-level team tables into one master table (teamdata_all)
# ============================================================

import os
import sqlite3
import pandas as pd
from core.config import DB_PATH
from core.log_config import setup_logger
from core.exceptions import PipelineError

logger = setup_logger("merge_team_data")


def merge_team_data():
    """Merge all season-level team tables into one master table (teamdata_all)."""
    if not os.path.exists(DB_PATH):
        raise PipelineError(f"Database not found at {DB_PATH}")

    try:
        con = sqlite3.connect(DB_PATH)
        cursor = con.cursor()

        # Get list of all season tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'teamdata_%'")
        season_tables = [row[0] for row in cursor.fetchall()]
        logger.info(f"Found {len(season_tables)} season tables: {season_tables}")

        if not season_tables:
            raise PipelineError("No season tables found in database.")

        # Merge all season tables into one DataFrame
        frames = []
        for table in season_tables:
            try:
                df = pd.read_sql_query(f"SELECT * FROM {table}", con)
                df["Season"] = table.replace("teamdata_", "")
                frames.append(df)
                logger.info(f"Loaded {len(df)} rows from {table}")
            except Exception as e:
                logger.warning(f"Skipping {table} due to error: {e}")

        if not frames:
            raise PipelineError("No valid season data frames to merge.")

        master_df = pd.concat(frames, ignore_index=True)

        # Save to master table
        master_df.to_sql("teamdata_all", con, if_exists="replace", index=False)
        logger.info(f"✅ Merged {len(season_tables)} tables into teamdata_all with {len(master_df)} rows")

    except Exception as e:
        logger.error(f"❌ Failed to merge team data: {e}")
        raise PipelineError(f"Merge failed: {e}")
    finally:
        con.close()


if __name__ == "__main__":
    merge_team_data()