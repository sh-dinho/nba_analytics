# ============================================================
# File: src/scripts/generate_today_schedule.py
# Purpose: Generate today's NBA schedule for predictions
# Project: nba_analysis
# Version: 1.0
# ============================================================

import os
import pandas as pd
import logging
from src.api.nba_api_wrapper import fetch_today_games

# -----------------------------
# CONFIG
# -----------------------------
CACHE_DIR = "data/cache"
SCHEDULE_FILE = os.path.join(CACHE_DIR, "schedule.parquet")

os.makedirs(CACHE_DIR, exist_ok=True)

# -----------------------------
# LOGGING
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# -----------------------------
# MAIN
# -----------------------------
def main():
    logging.info("Generating today's NBA schedule...")

    today_games = fetch_today_games()
    if today_games.empty:
        logging.warning("No NBA games today. Schedule file will not be created.")
        return

    # Keep relevant columns for the pipeline
    schedule_df = today_games[['GAME_DATE', 'home_team', 'away_team', 'TEAM_NAME', 'MATCHUP']].copy()

    # Add unique GAME_ID (home_vs_away_date)
    schedule_df['GAME_ID'] = (
        schedule_df['away_team'] + "_vs_" + schedule_df['home_team'] + "_" +
        schedule_df['GAME_DATE'].dt.strftime("%Y%m%d")
    )

    # Save as parquet for daily pipeline
    schedule_df.to_parquet(SCHEDULE_FILE, index=False)
    logging.info(f"Schedule saved to {SCHEDULE_FILE}. Total games: {len(schedule_df)}")


if __name__ == "__main__":
    main()
