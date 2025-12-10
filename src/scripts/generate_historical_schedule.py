# ============================================================
# File: src/scripts/generate_historical_schedule.py
# Purpose: Fetch historical NBA games and save as parquet
# ============================================================

import pandas as pd
import logging
from nba_api.stats.endpoints import leaguegamefinder
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SEASONS = ["2022-23", "2023-24", "2024-25"]  # Add more as needed
OUTPUT_FILE = "data/cache/historical_schedule.parquet"

all_games = []

logging.info("Fetching historical NBA games...")
for season in SEASONS:
    try:
        logging.info(f"Fetching season {season}...")
        gamefinder = leaguegamefinder.LeagueGameFinder(season_nullable=season)
        df = gamefinder.get_data_frames()[0]
        # Keep relevant columns
        df = df[['GAME_DATE', 'TEAM_NAME', 'MATCHUP']]
        all_games.append(df)
        logging.info(f"Season {season}: {len(df)} games fetched")
    except Exception as e:
        logging.error(f"Error fetching season {season}: {e}")

if all_games:
    combined_df = pd.concat(all_games, ignore_index=True)
    # Remove duplicates
    combined_df = combined_df.drop_duplicates(subset=['GAME_DATE', 'TEAM_NAME', 'MATCHUP'])
    combined_df['GAME_DATE'] = pd.to_datetime(combined_df['GAME_DATE'])
    combined_df.to_parquet(OUTPUT_FILE, index=False)
    logging.info(f"Historical schedule saved to {OUTPUT_FILE} ({len(combined_df)} games total)")
else:
    logging.warning("No games fetched. Historical schedule not created.")
