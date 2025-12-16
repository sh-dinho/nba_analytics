# ============================================================
# File: src/scripts/update_historical.py
# Purpose: Incrementally update NBA historical schedule
# ============================================================

import pandas as pd
from pathlib import Path
import logging
from datetime import datetime, timedelta
import requests
import json

RAW_DIR = Path("data/raw")
CACHE_DIR = Path("data/cache")
HISTORY_DIR = Path("data/history")
HISTORY_FILE = CACHE_DIR / "master_schedule.parquet"

SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]  # adjust as needed
NBA_NET_URL = "https://data.nba.net/prod/v2/{season}/schedule.json"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


def load_existing_schedule():
    if HISTORY_FILE.exists():
        df = pd.read_parquet(HISTORY_FILE)
        logging.info(f"Loaded existing historical schedule (rows={len(df)})")
        return df
    return pd.DataFrame()


def save_schedule(df: pd.DataFrame):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(HISTORY_FILE, index=False)
    logging.info(f"Updated historical schedule saved (rows={len(df)})")


def fetch_season_schedule(season: str) -> pd.DataFrame:
    url = NBA_NET_URL.format(season=season)
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        games = data.get("league", {}).get("standard", [])
        df = pd.json_normalize(games)
        # Save raw JSON
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        json_file = RAW_DIR / f"{season}_schedule.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(games, f, indent=2)
        # Save Parquet
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        parquet_file = HISTORY_DIR / f"{season}_schedule.parquet"
        df.to_parquet(parquet_file, index=False)
        logging.info(f"Fetched {len(df)} games for season {season}")
        return df
    except Exception as e:
        logging.error(f"Failed to fetch season {season}: {e}")
        return pd.DataFrame()


def update_historical_schedule():
    existing = load_existing_schedule()
    all_data = []

    for season in SEASONS:
        # Load local parquet first if exists
        parquet_file = HISTORY_DIR / f"{season}_schedule.parquet"
        if parquet_file.exists():
            df_season = pd.read_parquet(parquet_file)
            logging.info(f"Loaded local {season} parquet ({len(df_season)} rows)")
        else:
            df_season = fetch_season_schedule(season)

        # Identify missing dates
        if not existing.empty:
            existing_dates = pd.to_datetime(existing["startDate"])
            season_dates = pd.to_datetime(df_season["startDate"])
            missing_dates = season_dates[~season_dates.isin(existing_dates)]
            df_season = df_season[df_season["startDate"].isin(missing_dates)]
            logging.info(f"{len(df_season)} missing games to append for {season}")

        all_data.append(df_season)

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        if not combined.empty:
            updated = pd.concat([existing, combined], ignore_index=True)
            save_schedule(updated)
        else:
            logging.info("No new games to append. Historical schedule is up-to-date.")
    else:
        logging.warning("No historical data found to update.")


if __name__ == "__main__":
    logging.info("===== INCREMENTAL HISTORICAL UPDATE START =====")
    update_historical_schedule()
    logging.info("===== INCREMENTAL HISTORICAL UPDATE END =====")
