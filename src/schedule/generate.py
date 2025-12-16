# ============================================================
# File: src/schedule/generate.py
# Purpose: Download historical NBA schedules
# ============================================================

import logging
from pathlib import Path
import requests
import pandas as pd

logger = logging.getLogger(__name__)

RAW_PATH = Path("data/raw")
HISTORY_PATH = Path("data/history")
RAW_PATH.mkdir(parents=True, exist_ok=True)
HISTORY_PATH.mkdir(parents=True, exist_ok=True)


def fetch_season_schedule(season_year: int) -> pd.DataFrame:
    """
    Fetch the NBA season schedule for a given year.
    Returns a DataFrame with games.
    """
    url = f"https://data.nba.net/prod/v2/{season_year}/schedule.json"
    try:
        r = requests.get(url, timeout=10, verify=True)
        r.raise_for_status()
        data = r.json()
        games = data.get("league", {}).get("standard", [])
        df = pd.json_normalize(games)
        logger.info(f"Downloaded {len(df)} games for season {season_year}")
        return df
    except Exception as e:
        logger.error(f"Failed to fetch season {season_year}: {e}")
        return pd.DataFrame()


def generate_historical(config) -> pd.DataFrame:
    """
    Download historical seasons (2022-present) and save to local parquet.
    """
    all_seasons = []
    for year in range(2022, 2026):  # 2022-23 to 2025-26
        df_season = fetch_season_schedule(year)
        if not df_season.empty:
            all_seasons.append(df_season)

    if all_seasons:
        historical_df = pd.concat(all_seasons, ignore_index=True)
        HISTORY_PATH.mkdir(parents=True, exist_ok=True)
        historical_file = HISTORY_PATH / "historical_schedule.parquet"
        historical_df.to_parquet(historical_file, index=False)
        logger.info(f"Saved historical schedule (rows={len(historical_df)})")
        return historical_df
    else:
        logger.error("No historical schedule downloaded.")
        return pd.DataFrame()
