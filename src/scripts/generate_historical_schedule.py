#!/usr/bin/env python
# ============================================================
# File: src/scripts/generate_historical_schedule.py
# Purpose: Fetch historical NBA games across multiple seasons and save as Parquet + CSV
# ============================================================

import os
import sys
import time
import pandas as pd
import requests
from nba_api.stats.endpoints import leaguegamefinder
from src.utils.logging_config import configure_logging
from src.schemas import HISTORICAL_SCHEDULE_COLUMNS

SEASONS = ["2022-23", "2023-24", "2024-25"]
OUTPUT_FILE = "data/cache/historical_schedule.parquet"
CSV_BACKUP = "data/cache/historical_schedule.csv"


def ensure_dir(path: str):
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)


def fetch_season(season: str, retries: int = 3, delay: int = 10) -> pd.DataFrame:
    logger = configure_logging(name="scripts.generate_historical_schedule")
    for attempt in range(retries):
        try:
            logger.info(
                "Fetching data for season %s (attempt %d)...", season, attempt + 1
            )
            gamefinder = leaguegamefinder.LeagueGameFinder(season_nullable=season)
            df = gamefinder.get_data_frames()[0]

            # Keep relevant columns and add season
            keep = [
                c
                for c in [
                    "GAME_DATE",
                    "TEAM_NAME",
                    "MATCHUP",
                    "GAME_ID",
                    "TEAM_ID",
                    "PTS",
                    "WL",
                ]
                if c in df.columns
            ]
            df = df[keep]
            df["SEASON"] = season
            return df
        except (
            requests.exceptions.ReadTimeout,
            requests.exceptions.ConnectionError,
        ) as e:
            logger.warning(
                "Network error fetching season %s: %s. Retrying in %s seconds...",
                season,
                e,
                delay,
            )
            time.sleep(delay)
        except Exception as e:
            logger.error("Error fetching season %s: %s", season, e)
            return pd.DataFrame()
    return pd.DataFrame()


def main():
    logger = configure_logging(name="scripts.generate_historical_schedule")
    ensure_dir(OUTPUT_FILE)

    all_games = []
    logger.info("Fetching historical NBA games...")

    for season in SEASONS:
        df = fetch_season(season)
        if not df.empty:
            all_games.append(df)
            logger.info("Season %s: %d games fetched", season, len(df))

    if not all_games:
        logger.warning(
            "No games fetched. Writing empty historical schedule with schema."
        )
        empty = pd.DataFrame(columns=HISTORICAL_SCHEDULE_COLUMNS)
        empty.to_parquet(OUTPUT_FILE, index=False)
        empty.to_csv(CSV_BACKUP, index=False)
        sys.exit(0)

    combined_df = pd.concat(all_games, ignore_index=True)

    # Deduplicate by GAME_ID + TEAM_ID to keep both teams per game
    initial_len = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=["GAME_ID", "TEAM_ID"])
    final_len = len(combined_df)

    # Convert GAME_DATE to datetime
    combined_df["GAME_DATE"] = pd.to_datetime(combined_df["GAME_DATE"], errors="coerce")

    # Ensure PTS_OPP exists (optional; may be filled later by boxscores)
    if "PTS_OPP" not in combined_df.columns:
        combined_df["PTS_OPP"] = None

    logger.info("Removed %d duplicate rows.", initial_len - final_len)
    logger.info("Final dataset contains %d unique rows.", final_len)

    try:
        combined_df.to_parquet(OUTPUT_FILE, index=False)
        combined_df.to_csv(CSV_BACKUP, index=False)
        logger.info(
            "Historical schedule saved to %s and %s (%d rows total)",
            OUTPUT_FILE,
            CSV_BACKUP,
            final_len,
        )
    except Exception as e:
        logger.error("Error saving historical schedule: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
