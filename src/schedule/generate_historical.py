# ============================================================
# File: src/schedule/generate_historical.py
# Purpose: Download historical NBA schedules
# ============================================================

import pandas as pd
import requests
from pathlib import Path
from loguru import logger
from datetime import datetime, timedelta

RAW_DIR = Path("data/raw")
PARQUET_DIR = Path("data/history")
RAW_DIR.mkdir(parents=True, exist_ok=True)
PARQUET_DIR.mkdir(parents=True, exist_ok=True)


def fetch_season_schedule(year: int) -> pd.DataFrame:
    """
    Fetch full season schedule from NBA data API for a given year.
    """
    url = f"https://data.nba.net/prod/v2/{year}/schedule.json"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        games = data.get("league", {}).get("standard", [])
        df = pd.json_normalize(games)
        df["SEASON"] = year
        return df
    except Exception as e:
        logger.error(f"Failed to fetch season {year}: {e}")
        return pd.DataFrame()


def fetch_games_by_date(date: pd.Timestamp) -> pd.DataFrame:
    """
    Fetch games for a specific date.
    """
    year = date.year
    month = str(date.month).zfill(2)
    day = str(date.day).zfill(2)
    url = f"https://data.nba.net/prod/v1/{year}{month}{day}/scoreboard.json"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        games = data.get("games", [])
        df = pd.json_normalize(games)
        if not df.empty:
            df["GAME_DATE"] = date
        return df
    except Exception as e:
        logger.error(f"Failed to fetch games for {date.date()}: {e}")
        return pd.DataFrame()


def generate_historical(dates: pd.DatetimeIndex = None) -> pd.DataFrame:
    """
    Generate historical schedule.
    If dates is None, fetch all seasons 2022-2026.
    If dates are provided, fetch only games for those dates.
    """
    all_data = []

    if dates is None:
        logger.info("Fetching full seasons 2022-2026...")
        for year in range(2022, 2026):
            df = fetch_season_schedule(year)
            if not df.empty:
                all_data.append(df)
            logger.info(f"Fetched {df.shape[0]} games for season {year}")
    else:
        logger.info(f"Fetching games for {len(dates)} missing days...")
        for date in dates:
            df = fetch_games_by_date(date)
            if not df.empty:
                all_data.append(df)
            logger.info(f"Fetched {df.shape[0]} games for {date.date()}")

    if all_data:
        full_df = pd.concat(all_data, ignore_index=True).drop_duplicates(
            subset=["gameId"]
        )
        logger.info(f"Total historical games fetched: {full_df.shape[0]}")
        return full_df
    else:
        logger.warning("No historical games fetched.")
        return pd.DataFrame()
