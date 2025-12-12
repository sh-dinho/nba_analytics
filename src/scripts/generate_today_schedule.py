# ============================================================
# File: src/scripts/generate_today_schedule.py
# Purpose: Generate today's NBA schedule (or next game day) and save to CSV/Parquet
# ============================================================

import logging
import os
import sys
import pandas as pd
from src.api.nba_api_client import fetch_today_games
from src.schemas import TODAY_SCHEDULE_COLUMNS, normalize_today_schedule

logger = logging.getLogger("scripts.generate_today_schedule")
logging.basicConfig(level=logging.INFO)


def main():
    logger.info("Fetching today's NBA schedule...")

    try:
        df = fetch_today_games()
    except Exception as e:
        logger.error("Error fetching today's games: %s", e)
        sys.exit(1)

    out_path = "data/results/today_schedule.csv"
    parquet_path = "data/results/today_schedule.parquet"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if df.empty:
        logger.warning("No games found today or in the next 7 days.")
        pd.DataFrame(columns=TODAY_SCHEDULE_COLUMNS).to_csv(out_path, index=False)
        pd.DataFrame(columns=TODAY_SCHEDULE_COLUMNS).to_parquet(
            parquet_path, index=False
        )
        logger.info("Empty schedule files written to %s and %s", out_path, parquet_path)
        sys.exit(0)

    df = normalize_today_schedule(df)

    try:
        df.to_csv(out_path, index=False)
        df.to_parquet(parquet_path, index=False)
        logger.info(
            "Schedule saved to %s and %s (shape: %s)", out_path, parquet_path, df.shape
        )
    except Exception as e:
        logger.error("Error saving schedule files: %s", e)
        sys.exit(1)

    if "GAME_DATE_EST" in df.columns and "GAME_TYPE" in df.columns:
        next_date = df["GAME_DATE_EST"].iloc[0]
        game_type = df["GAME_TYPE"].iloc[0]
        logger.info("Next NBA game day is %s (%s)", next_date, game_type)
    else:
        logger.warning(
            "GAME_DATE_EST or GAME_TYPE column missing. Cannot display next game day info."
        )


if __name__ == "__main__":
    main()
