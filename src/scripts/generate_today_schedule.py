# ============================================================
# File: src/scripts/generate_today_schedule.py
# Purpose: Generate today's NBA schedule (or next game day) and save to CSV
# ============================================================

import logging
import os
import pandas as pd
from src.api.nba_api_client import fetch_today_games

logger = logging.getLogger("scripts.generate_today_schedule")
logging.basicConfig(level=logging.INFO)


def main():
    logger.info("Fetching today's NBA schedule...")
    df = fetch_today_games()

    if df.empty:
        logger.warning("No games found today or in the next 7 days.")
        return

    # Ensure results folder exists
    out_path = "data/results/today_schedule.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Save to CSV
    df.to_csv(out_path, index=False)
    logger.info("Schedule saved to %s (shape: %s)", out_path, df.shape)

    # Optional: also save to parquet for consistency
    parquet_path = "data/results/today_schedule.parquet"
    df.to_parquet(parquet_path, index=False)
    logger.info("Schedule also saved to %s", parquet_path)

    # Display next game day info
    next_date = df["GAME_DATE_EST"].iloc[0]
    game_type = df["GAME_TYPE"].iloc[0]
    logger.info("Next NBA game day is %s (%s)", next_date, game_type)


if __name__ == "__main__":
    main()
