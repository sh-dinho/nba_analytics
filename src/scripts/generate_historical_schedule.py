#!/usr/bin/env python
# ============================================================
# File: src/scripts/generate_historical_schedule.py
# Purpose: Fetch historical NBA games across multiple seasons and save as Parquet
# Project: nba_analysis
# Version: 1.3 (named logger, safer deduplication, exit codes)
# ============================================================

import os
import sys
import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder
from src.utils.logging_config import configure_logging

# -----------------------------
# CONFIGURATION
# -----------------------------
SEASONS = ["2022-23", "2023-24", "2024-25"]
OUTPUT_FILE = "data/cache/historical_schedule.parquet"


def ensure_dir(path: str):
    """Ensure directory exists for a given file path."""
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)


# -----------------------------
# MAIN
# -----------------------------
def main():
    logger = configure_logging(name="scripts.generate_historical_schedule")
    ensure_dir(OUTPUT_FILE)

    all_games = []
    logger.info("Fetching historical NBA games...")

    for season in SEASONS:
        try:
            logger.info("Fetching data for season %s...", season)
            gamefinder = leaguegamefinder.LeagueGameFinder(season_nullable=season)
            df = gamefinder.get_data_frames()[0]

            # Keep relevant columns
            df = df[
                ["GAME_DATE", "TEAM_NAME", "MATCHUP", "GAME_ID", "TEAM_ID", "PTS", "WL"]
            ]
            all_games.append(df)
            logger.info("Season %s: %d games fetched", season, len(df))
        except Exception as e:
            logger.error("Error fetching season %s: %s", season, e)

    if not all_games:
        logger.warning("No games fetched. Historical schedule not created.")
        sys.exit(0)

    combined_df = pd.concat(all_games, ignore_index=True)

    # Deduplicate by GAME_ID + TEAM_ID to keep both teams per game
    initial_len = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=["GAME_ID", "TEAM_ID"])
    final_len = len(combined_df)

    # Convert GAME_DATE to datetime
    combined_df["GAME_DATE"] = pd.to_datetime(combined_df["GAME_DATE"], errors="coerce")

    logger.info("Removed %d duplicate rows.", initial_len - final_len)
    logger.info("Final dataset contains %d unique rows.", final_len)

    try:
        combined_df.to_parquet(OUTPUT_FILE, index=False)
        logger.info(
            "Historical schedule saved to %s (%d rows total)", OUTPUT_FILE, final_len
        )
    except Exception as e:
        logger.error("Error saving historical schedule: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
