# ============================================================
# File: scripts/fetch_features.py
# Purpose: Fetch NBA team stats from the NBA API and prepare features
# ============================================================

import requests
import pandas as pd
from typing import Dict, Any
from scripts.utils import clean_team_name  # optional: for consistent team names

def fetch_nba_team_stats(api_url: str) -> pd.DataFrame:
    """
    Fetch team stats from NBA API and return a DataFrame.

    Args:
        api_url: Fully formatted NBA API URL.

    Returns:
        DataFrame with team stats ready for feature engineering.
    """
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
        "Referer": "https://www.nba.com/"
    }

    response = requests.get(api_url, headers=headers, timeout=10)
    response.raise_for_status()  # Raise exception if request fails

    data_json = response.json()
    # NBA API returns resultSets; find "TeamStats" or first resultSet
    result_sets = data_json.get("resultSets", [])
    if not result_sets:
        raise ValueError("No resultSets found in NBA API response")

    # Extract headers and rows
    headers = result_sets[0]["headers"]
    rows = result_sets[0]["rowSet"]

    df = pd.DataFrame(rows, columns=headers)

    # Optional: normalize team names
    if "TEAM_NAME" in df.columns:
        df["TEAM_NAME"] = df["TEAM_NAME"].apply(clean_team_name)

    return df
