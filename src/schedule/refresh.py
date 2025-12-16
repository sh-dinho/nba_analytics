# ============================================================
# File: src/schedule/incremental.py
# Purpose: Refresh master schedule incrementally
# ============================================================

import logging
import pandas as pd

logger = logging.getLogger(__name__)


def refresh_incremental(today_schedule, config):
    master_file = config.schedule.master_file
    try:
        master_schedule = pd.read_parquet(master_file)
        updated = pd.concat([master_schedule, today_schedule]).drop_duplicates(
            subset="GAME_ID"
        )
        logger.info(f"Master schedule updated. Total rows: {len(updated)}")
    except FileNotFoundError:
        updated = today_schedule
        logger.info("No master schedule found. Using today's schedule as master.")

    master_file.parent.mkdir(parents=True, exist_ok=True)
    updated.to_parquet(master_file, index=False)
    return updated
