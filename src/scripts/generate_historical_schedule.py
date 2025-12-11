# ============================================================
# File: src/scripts/generate_historical_schedule.py
# Purpose: Fetch historical NBA games across multiple seasons and save as Parquet
# Project: nba_analysis
# Version: 1.1 (adds dependencies section, clarifies logging and output handling)
#
# Dependencies:
# - logging (standard library)
# - os (standard library)
# - datetime (standard library)
# - pandas
# - nba_api.stats.endpoints.leaguegamefinder
# ============================================================

import logging
import os
from datetime import datetime

import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder

# -----------------------------
# CONFIGURATION
# -----------------------------
SEASONS = ["2022-23", "2023-24", "2024-25"]  # Update as required, could be dynamic
OUTPUT_FILE = "data/cache/historical_schedule.parquet"
OUTPUT_DIR = os.path.dirname(OUTPUT_FILE)

# -----------------------------
# LOGGING CONFIGURATION
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# DATA FETCHING & PROCESSING
# -----------------------------
all_games = []
logging.info("Fetching historical NBA games...")

for season in SEASONS:
    try:
        logging.info(f"Fetching data for season {season}...")
        gamefinder = leaguegamefinder.LeagueGameFinder(season_nullable=season)
        df = gamefinder.get_data_frames()[0]

        # Keep relevant columns and log number of games
        df = df[["GAME_DATE", "TEAM_NAME", "MATCHUP"]]
        all_games.append(df)
        logging.info(f"Season {season}: {len(df)} games fetched")

    except Exception as e:
        logging.error(f"Error fetching season {season}: {e}")

# -----------------------------
# COMBINE & CLEAN DATA
# -----------------------------
if all_games:
    combined_df = pd.concat(all_games, ignore_index=True)

    # Remove duplicates based on GAME_DATE, TEAM_NAME, and MATCHUP
    initial_len = len(combined_df)
    combined_df = combined_df.drop_duplicates(
        subset=["GAME_DATE", "TEAM_NAME", "MATCHUP"]
    )
    final_len = len(combined_df)

    # Convert GAME_DATE to datetime format
    combined_df["GAME_DATE"] = pd.to_datetime(combined_df["GAME_DATE"])

    logging.info(f"Removed {initial_len - final_len} duplicate games.")
    logging.info(f"Final dataset contains {final_len} unique games.")

    # Save to Parquet
    try:
        combined_df.to_parquet(OUTPUT_FILE, index=False)
        logging.info(
            f"Historical schedule saved to {OUTPUT_FILE} ({final_len} games total)"
        )
    except Exception as e:
        logging.error(f"Error saving historical schedule: {e}")
else:
    logging.warning("No games fetched. Historical schedule not created.")
