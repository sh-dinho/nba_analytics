# ============================================================
# File: src/scripts/download_baseline_schedule.py
# Purpose: Download full NBA season schedule from Basketball-Reference or GitHub
# ============================================================

import logging
import os
import sys
import pandas as pd
import requests
from bs4 import BeautifulSoup
from src.schemas import HISTORICAL_SCHEDULE_COLUMNS, normalize_df

logger = logging.getLogger("scripts.download_baseline_schedule")
logging.basicConfig(level=logging.INFO)

BASELINE_PATH = "data/reference/season_schedule.csv"


def fetch_schedule_bref(season: str = "2025"):
    """
    Fetch NBA schedule from Basketball-Reference for a given season.
    Example season: "2025" (for 2025-26 season).
    """
    url = f"https://www.basketball-reference.com/leagues/NBA_{season}_games.html"
    logger.info("Fetching schedule from %s", url)

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        logger.error("Failed to fetch schedule: %s", e)
        return pd.DataFrame()

    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", {"id": "schedule"})
    if table is None:
        logger.error("No schedule table found on Basketball-Reference.")
        return pd.DataFrame()

    df = pd.read_html(str(table))[0]

    # Drop header rows that repeat
    df = df[df["Date"] != "Date"]

    # Normalize columns
    df = df.rename(
        columns={
            "Date": "GAME_DATE",
            "Visitor/Neutral": "AWAY_TEAM",
            "Home/Neutral": "HOME_TEAM",
            "PTS": "PTS",
            "PTS.1": "PTS_OPP",
        }
    )

    # Add GAME_ID placeholder (can be generated later)
    df["GAME_ID"] = df.index.astype(str)
    df["WL"] = None
    df["SEASON"] = season

    # Ensure schema consistency
    df = normalize_df(df, HISTORICAL_SCHEDULE_COLUMNS)
    return df


def main():
    os.makedirs(os.path.dirname(BASELINE_PATH), exist_ok=True)

    df = fetch_schedule_bref("2025")
    if df.empty:
        logger.error("No schedule downloaded. Aborting.")
        sys.exit(1)

    df.to_csv(BASELINE_PATH, index=False)
    logger.info("Baseline schedule saved to %s (%d rows)", BASELINE_PATH, len(df))


if __name__ == "__main__":
    main()
