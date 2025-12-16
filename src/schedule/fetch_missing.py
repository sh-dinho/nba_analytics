import logging
import requests
import pandas as pd
from pathlib import Path

DATA_HISTORY = Path("data/history")


def fetch_missing(master_schedule: pd.DataFrame, seasons: list):
    """
    Download missing historical games for specified seasons.
    Only downloads seasons that are missing.
    Saves both raw JSON and Parquet.
    """
    DATA_HISTORY.mkdir(parents=True, exist_ok=True)

    for season in seasons:
        if not master_schedule.empty and season in master_schedule["season"].unique():
            logging.info(f"Season {season} already exists locally, skipping download")
            continue

        logging.info(f"Fetching season {season}")
        try:
            year_start = season.split("-")[0]
            url = f"https://data.nba.net/prod/v2/{year_start}/schedule.json"
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            games = data.get("league", {}).get("standard", [])
            if not games:
                logging.warning(f"No games found for season {season}")
                continue

            df = pd.json_normalize(games)
            df["season"] = season

            # Save raw JSON
            raw_file = DATA_HISTORY / f"{season}.json"
            with open(raw_file, "w", encoding="utf-8") as f:
                f.write(resp.text)

            # Save Parquet
            parquet_file = DATA_HISTORY / f"{season}.parquet"
            df.to_parquet(parquet_file, index=False)
            logging.info(f"Saved {len(df)} games for {season} (Parquet + JSON)")
        except requests.RequestException as e:
            logging.error(f"Failed to fetch season {season}: {e}")
