# ============================================================
# File: scripts/fetch_season_data.py
# Purpose: Automate NBA season data ingestion using config.toml
# ============================================================

import toml
import requests
import os
import pandas as pd

CONFIG_PATH = "config.toml"
OUTPUT_DIR = "data/seasons"

def build_url(template, start_date, end_date, start_year, end_year, season_label):
    """
    Fill placeholders in data_url template:
    {0} = end month
    {1} = end day
    {2} = start year
    {3} = end year
    {4} = season_label
    """
    end_month, end_day, end_year_str = end_date.split("-")[1], end_date.split("-")[2], end_date.split("-")[0]
    return template.format(end_month, end_day, start_year, end_year, season_label)

def fetch_season_data():
    config = toml.load(CONFIG_PATH)
    url_template = config["data_url"]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for season, params in config["get-data"].items():
        season_label = params["season_label"]
        start_date = params["start_date"]
        end_date = params["end_date"]
        start_year = params["start_year"]
        end_year = params["end_year"]

        url = build_url(url_template, start_date, end_date, start_year, end_year, season_label)
        print(f"Fetching {season_label} → {url}")

        # NBA stats API requires headers
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://www.nba.com/",
            "Accept": "application/json"
        }

        r = requests.get(url, headers=headers)
        if r.status_code == 200:
            data = r.json()
            # Extract rows
            rows = data["resultSets"][0]["rowSet"]
            headers = data["resultSets"][0]["headers"]
            df = pd.DataFrame(rows, columns=headers)

            out_path = os.path.join(OUTPUT_DIR, f"teamdata_{season_label}.csv")
            df.to_csv(out_path, index=False)
            print(f"✅ Saved {season_label} data to {out_path}")
        else:
            print(f"❌ Failed to fetch {season_label}: {r.status_code}")

if __name__ == "__main__":
    fetch_season_data()