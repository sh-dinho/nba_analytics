#!/usr/bin/env python
# ============================================================
# File: src/scripts/generate_today_schedule.py
# Purpose: Generate today's NBA schedule for predictions
# Project: nba_analysis
# Version: 1.4 (named logger, safe schema, exit codes)
# ============================================================

import os
import sys
import pandas as pd
from src.utils.logging_config import configure_logging
from src.utils.nba_api_wrapper import fetch_today_games
from src.utils.mapping import map_team_ids

# -----------------------------
# CONFIG
# -----------------------------
CACHE_DIR = "data/cache"
SCHEDULE_FILE = os.path.join(CACHE_DIR, "schedule.parquet")


def ensure_dir(path: str):
    """Ensure directory exists for a given file path."""
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)


# -----------------------------
# MAIN FUNCTION
# -----------------------------
def main():
    logger = configure_logging(name="scripts.generate_today_schedule")
    logger.info("Generating today's NBA schedule...")

    try:
        today_games = fetch_today_games()

        if today_games is None or today_games.empty:
            logger.warning("No NBA games today. Schedule file will not be created.")
            sys.exit(0)

        # Map team IDs to names for readability
        schedule_df = map_team_ids(today_games, team_col="HOME_TEAM_ID").rename(
            columns={"TEAM_NAME": "HOME_TEAM_NAME"}
        )
        schedule_df = map_team_ids(schedule_df, team_col="AWAY_TEAM_ID").rename(
            columns={"TEAM_NAME": "AWAY_TEAM_NAME"}
        )

        # Keep relevant columns
        cols = [
            "GAME_ID",
            "HOME_TEAM_ID",
            "AWAY_TEAM_ID",
            "HOME_TEAM_NAME",
            "AWAY_TEAM_NAME",
        ]
        schedule_df = schedule_df[cols].copy()

        # Log the number of games found and processed
        logger.info("Total games found: %d", len(schedule_df))
        logger.debug("Game IDs: %s", schedule_df["GAME_ID"].tolist())

        # Ensure directory exists
        ensure_dir(SCHEDULE_FILE)

        # Save the schedule to a Parquet file
        schedule_df.to_parquet(SCHEDULE_FILE, index=False)
        logger.info(
            "Schedule successfully saved to %s. Total games: %d",
            SCHEDULE_FILE,
            len(schedule_df),
        )

    except Exception as e:
        logger.error("An error occurred while generating today's schedule: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
