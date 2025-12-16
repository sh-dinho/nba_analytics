# ============================================================
# File: src/scripts/download_offline_historical.py
# Purpose: Download historical NBA schedules and save locally
# ============================================================

import logging
from pathlib import Path
import pandas as pd
import requests

DATA_RAW = Path("data/raw")
DATA_CACHE = Path("data/cache")
DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_CACHE.mkdir(parents=True, exist_ok=True)

SEASONS = ["2022", "2023", "2024", "2025"]


def download_season(season: str) -> pd.DataFrame:
    url = f"https://data.nba.net/prod/v2/{season}/schedule.json"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        games = []
        for game in data.get("league", {}).get("standard", []):
            games.append(
                {
                    "game_id": game["gameId"],
                    "season": season,
                    "home_team": game["hTeam"]["teamId"],
                    "away_team": game["vTeam"]["teamId"],
                    "home_score": int(game.get("hTeam", {}).get("score", 0)),
                    "away_score": int(game.get("vTeam", {}).get("score", 0)),
                    "date": game["startDateEastern"],
                }
            )
        df = pd.DataFrame(games)
        return df
    except Exception as e:
        logging.error(f"Failed to fetch season {season}: {e}")
        return pd.DataFrame()


def download_missing_seasons(master_schedule: pd.DataFrame) -> pd.DataFrame:
    """
    Check existing data, download only missing seasons, and combine
    """
    combined = master_schedule.copy()
    for season in SEASONS:
        if not master_schedule.empty and season in master_schedule["season"].values:
            logging.info(f"Season {season} already exists, skipping download")
            continue
        logging.info(f"Fetching season {season}")
        df = download_season(season)
        if df.empty:
            logging.warning(f"No data fetched for season {season}")
            continue
        # Remove duplicates
        df = pd.concat([combined, df], ignore_index=True).drop_duplicates(
            subset=["game_id"]
        )
        combined = df
    return combined


def fetch_games_by_date(date_str: str) -> pd.DataFrame:
    """
    Fetch games for a specific date (YYYY-MM-DD) from NBA API
    """
    url = f"https://data.nba.net/prod/v2/{date_str}/scoreboard.json"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        games = []
        for game in data.get("games", []):
            games.append(
                {
                    "game_id": game["gameId"],
                    "season": game["seasonStageId"],
                    "home_team": game["hTeam"]["teamId"],
                    "away_team": game["vTeam"]["teamId"],
                    "home_score": int(game.get("hTeam", {}).get("score", 0)),
                    "away_score": int(game.get("vTeam", {}).get("score", 0)),
                    "date": date_str,
                }
            )
        return pd.DataFrame(games)
    except Exception as e:
        logging.error(f"Failed to fetch games for {date_str}: {e}")
        return pd.DataFrame()
