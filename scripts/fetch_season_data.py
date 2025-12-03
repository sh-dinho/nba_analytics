# ============================================================
# File: scripts/fetch_season_data.py
# Purpose: Automate NBA season data ingestion using config.toml
# ============================================================

import os
import toml
import requests
import pandas as pd
from core.config import BASE_DATA_DIR
from core.log_config import setup_logger
from core.exceptions import PipelineError, DataError
from core.utils import ensure_columns

CONFIG_PATH = "config.toml"
OUTPUT_DIR = os.path.join(BASE_DATA_DIR, "seasons")

logger = setup_logger("fetch_season_data")


def build_url(template: str, start_date: str, end_date: str,
              start_year: str, end_year: str, season_label: str) -> str:
    """
    Fill placeholders in data_url template:
    {0} = end month
    {1} = end day
    {2} = start year
    {3} = end year
    {4} = season_label
    """
    try:
        end_year_str, end_month, end_day = end_date.split("-")
    except ValueError:
        raise DataError(f"Invalid end_date format: {end_date}. Expected YYYY-MM-DD.")

    return template.format(end_month, end_day, start_year, end_year, season_label)


def fetch_season_data() -> None:
    """Fetch NBA season data based on config.toml settings."""
    try:
        config = toml.load(CONFIG_PATH)
    except Exception as e:
        raise PipelineError(f"Failed to load config file {CONFIG_PATH}: {e}")

    url_template = config.get("data_url")
    if not url_template:
        raise DataError("Missing 'data_url' in config.toml")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for season, params in config.get("get-data", {}).items():
        season_label = params.get("season_label")
        start_date = params.get("start_date")
        end_date = params.get("end_date")
        start_year = params.get("start_year")
        end_year = params.get("end_year")

        if not all([season_label, start_date, end_date, start_year, end_year]):
            logger.warning(f"⚠️ Skipping {season}: missing parameters in config.")
            continue

        url = build_url(url_template, start_date, end_date, start_year, end_year, season_label)
        logger.info(f"Fetching {season_label} → {url}")

        headers = {
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://www.nba.com/",
            "Accept": "application/json"
        }

        try:
            r = requests.get(url, headers=headers, timeout=30)
            r.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"❌ Failed to fetch {season_label}: {e}")
            continue

        try:
            data = r.json()
            rows = data["resultSets"][0]["rowSet"]
            headers = data["resultSets"][0]["headers"]
            df = pd.DataFrame(rows, columns=headers)

            ensure_columns(df, {"TEAM_ID", "TEAM_NAME"}, f"{season_label} season data")

            out_path = os.path.join(OUTPUT_DIR, f"teamdata_{season_label}.csv")
            df.to_csv(out_path, index=False)
            logger.info(f"✅ Saved {season_label} data to {out_path} ({len(df)} rows)")
        except Exception as e:
            logger.error(f"❌ Error processing {season_label} data: {e}")


if __name__ == "__main__":
    fetch_season_data()