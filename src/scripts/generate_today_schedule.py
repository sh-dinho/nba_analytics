# ============================================================
# File: src/scripts/generate_today_schedule.py
# Purpose: Generate today's NBA schedule for predictions
# Project: nba_analysis
# Version: 1.2 (adds dependencies section, clarifies logging and output handling)
#
# Dependencies:
# - logging (standard library)
# - os (standard library)
# - datetime (standard library)
# - pandas
# - src.api.nba_api_wrapper.fetch_today_games
# ============================================================

import logging
import os
from datetime import datetime

import pandas as pd
from src.api.nba_api_wrapper import fetch_today_games

# -----------------------------
# CONFIG
# -----------------------------
CACHE_DIR = "data/cache"
SCHEDULE_FILE = os.path.join(CACHE_DIR, "schedule.parquet")

# Ensure the cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

# -----------------------------
# LOGGING CONFIGURATION
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# -----------------------------
# MAIN FUNCTION
# -----------------------------
def main():
    logging.info("Generating today's NBA schedule...")

    try:
        # Fetch today's games from the API
        today_games = fetch_today_games()

        if today_games.empty:
            logging.warning("No NBA games today. Schedule file will not be created.")
            return

        # Process and clean the fetched data
        schedule_df = today_games[
            ["GAME_DATE", "home_team", "away_team", "TEAM_NAME", "MATCHUP"]
        ].copy()

        # Ensure GAME_DATE is in correct datetime format
        schedule_df["GAME_DATE"] = pd.to_datetime(schedule_df["GAME_DATE"], errors="coerce")

        # Add unique GAME_ID (home_vs_away_date)
        schedule_df["GAME_ID"] = (
            schedule_df["away_team"]
            + "_vs_"
            + schedule_df["home_team"]
            + "_"
            + schedule_df["GAME_DATE"].dt.strftime("%Y%m%d")
        )

        # Log the number of games found and processed
        logging.info(f"Total games found: {len(schedule_df)}")

        # Save the schedule to a Parquet file
        schedule_df.to_parquet(SCHEDULE_FILE, index=False)
        logging.info(f"Schedule successfully saved to {SCHEDULE_FILE}. Total games: {len(schedule_df)}")

    except Exception as e:
        logging.error(f"An error occurred while generating today's schedule: {e}")


if __name__ == "__main__":
    main()
