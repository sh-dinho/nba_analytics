#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NBA Historical Schedule Downloader and Processor

Author: Your Name
Date: 2023

Purpose:
    This script downloads NBA historical schedules for multiple seasons and stores them
    in Parquet format for further analysis. It uses asynchronous HTTP requests to fetch
    data from the NBA's data source and processes the results into a DataFrame for each season.

Dependencies:
    - pandas
    - aiohttp
    - asyncio
    - pathlib
    - loguru

Usage:
    - Use the `download_historical_schedule` function to download and store NBA schedules
      for a range of seasons.
    - Use the `load_historical_schedule` function to load previously downloaded data from
      Parquet files.
"""

import logging
import pandas as pd
from pathlib import Path
from loguru import logger
import asyncio
import aiohttp
from aiohttp import ClientSession

# -------------------------
# Logging setup
# -------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# -------------------------
# Paths & Configurations
# -------------------------
HISTORY_PATH = Path("data/history")
HISTORY_PATH.mkdir(parents=True, exist_ok=True)  # Ensure directory exists for saving data

# -------------------------
# Helper Functions
# -------------------------

async def fetch_season_schedule_async(year: int, session: ClientSession) -> pd.DataFrame:
    """
    Fetch the NBA season schedule for a given year asynchronously.

    Args:
        year (int): NBA season year (e.g., 2022, 2023).
        session (ClientSession): Asynchronous HTTP session.

    Returns:
        pd.DataFrame: A DataFrame with the fetched game details.
    """
    url = f"https://data.nba.net/prod/v2/{year}/schedule.json"
    try:
        async with session.get(url, timeout=10) as response:
            response.raise_for_status()  # Raise an error for HTTP request failures
            data = await response.json()

            games = []
            for game in data.get("league", {}).get("standard", []):
                games.append(
                    {
                        "game_id": game["gameId"],
                        "season": year,
                        "date": game["startDateEastern"],
                        "home_team": game["hTeam"]["teamId"],
                        "away_team": game["vTeam"]["teamId"],
                        "home_score": int(game["hTeam"].get("score", 0)),
                        "away_score": int(game["vTeam"].get("score", 0)),
                    }
                )

            df = pd.DataFrame(games)
            logger.info(f"Fetched {len(df)} games for season {year}")
            return df
    except Exception as e:
        logger.error(f"Failed to fetch season {year} schedule: {e}")
        return pd.DataFrame()


async def fetch_all_seasons_async(seasons: list) -> pd.DataFrame:
    """
    Fetch schedules for all provided seasons asynchronously.

    Args:
        seasons (list): List of seasons to fetch (e.g., [2022, 2023, 2024]).

    Returns:
        pd.DataFrame: Concatenated DataFrame containing the schedules for all seasons.
    """
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_season_schedule_async(season, session) for season in seasons]
        all_seasons_data = await asyncio.gather(*tasks)

    # Filter out any empty DataFrames and concatenate the rest
    all_seasons_data = [df for df in all_seasons_data if not df.empty]
    if all_seasons_data:
        historical_schedule = pd.concat(all_seasons_data, ignore_index=True)
        logger.info(f"Total historical schedule downloaded: {len(historical_schedule)} games")
        return historical_schedule
    else:
        logger.error("Failed to download any historical data.")
        return pd.DataFrame()


def download_historical_schedule(config=None, seasons=[2022, 2023, 2024, 2025]):
    """
    Download historical schedules for the defined seasons and save them to Parquet files.

    Args:
        config: Configuration object (optional).
        seasons (list): List of seasons to download.

    Returns:
        pd.DataFrame: Combined DataFrame with all downloaded historical data.
    """
    try:
        historical_schedule = asyncio.run(fetch_all_seasons_async(seasons))

        if historical_schedule.empty:
            logger.error("No data fetched for historical schedule.")
            return pd.DataFrame()

        # Save historical data to Parquet files
        for season in seasons:
            season_data = historical_schedule[historical_schedule["season"] == season]
            season_file = HISTORY_PATH / f"schedule_{season}.parquet"
            season_data.to_parquet(season_file, index=False)
            logger.info(f"Saved {len(season_data)} games for season {season} to {season_file}")

        return historical_schedule
    except Exception as e:
        logger.error(f"Error during download: {e}")
        return pd.DataFrame()


def load_historical_schedule() -> pd.DataFrame:
    """
    Load historical schedule from the saved Parquet files.

    Returns:
        pd.DataFrame: DataFrame containing all historical schedules.
    """
    season_files = list(HISTORY_PATH.glob("*.parquet"))
    if not season_files:
        logger.warning("No historical data found.")
        return pd.DataFrame()

    # Read and concatenate all Parquet files for each season
    dfs = [pd.read_parquet(f) for f in season_files]
    historical_schedule = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded historical schedule from {len(season_files)} files, total rows: {len(historical_schedule)}")
    return historical_schedule


# -------------------------
# Example Usage
# -------------------------
if __name__ == "__main__":
    # Fetch and store historical NBA schedules for the seasons 2022 to 2025
    seasons_to_fetch = [2022, 2023, 2024, 2025]
    historical_data = download_historical_schedule(seasons=seasons_to_fetch)

    # If historical data was downloaded, load it from Parquet files
    if not historical_data.empty:
        loaded_data = load_historical_schedule()
        print(loaded_data.head())  # Show first few rows of the loaded schedule data
