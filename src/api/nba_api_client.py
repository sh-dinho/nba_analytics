# ============================================================
# Path: src/api/nba_api_client.py
# Purpose: Unified NBA API client (JSON + nba_api.stats)
# Project: nba_analysis
# Version: 2.0 (caching, invalidation, stats wrapper)
# ============================================================

import os
import json
import logging
from datetime import datetime, timedelta
import pandas as pd
import requests
from typing import List
from nba_api.stats.endpoints import leaguegamefinder

# -----------------------------
# CONFIG
# -----------------------------
BASE_URL = "https://data.nba.com/data/v2015/json/mobile_teams/nba"
RAW_DIR = "data/raw"
CACHE_EXPIRY_HOURS = 24
SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]

EXPECTED_GAME_COLS = ["GAME_ID", "TEAM_ID", "OPPONENT_TEAM_ID", "date"]
EXPECTED_BOX_COLS = ["GAME_ID", "PLAYER_ID", "TEAM_ID", "PTS", "REB", "AST", "MIN"]

os.makedirs(RAW_DIR, exist_ok=True)

# -----------------------------
# CACHE HELPERS
# -----------------------------
def _cache_response(filename: str, data: dict):
    path = os.path.join(RAW_DIR, filename)
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logging.info(f"Cached API response: {path}")
    except Exception as e:
        logging.warning(f"Failed to cache response {filename}: {e}")

def _load_cache(filename: str) -> dict | None:
    path = os.path.join(RAW_DIR, filename)
    if os.path.exists(path):
        try:
            mtime = datetime.fromtimestamp(os.path.getmtime(path))
            if datetime.now() - mtime > timedelta(hours=CACHE_EXPIRY_HOURS):
                logging.info(f"Cache expired for {filename}, refreshing...")
                return None
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"Failed to load cache {filename}: {e}")
    return None

# -----------------------------
# JSON ENDPOINTS
# -----------------------------
def fetch_games(date: str, use_cache=True) -> pd.DataFrame:
    """Fetch NBA games for a given date from JSON API."""
    y, m, d = date.split("-")
    cache_file = f"schedule_{y}.json"

    data = _load_cache(cache_file) if use_cache else None
    if data is None:
        try:
            url = f"{BASE_URL}/{y}/league/00_full_schedule.json"
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            _cache_response(cache_file, data)
        except Exception as e:
            logging.error(f"Failed to fetch games for {date}: {e}")
            return pd.DataFrame(columns=EXPECTED_GAME_COLS)

    games = []
    for month in data.get("lscd", []):
        for game in month.get("mscd", {}).get("g", []):
            game_date = game.get("gdte")
            if game_date and datetime.strptime(game_date, "%Y-%m-%d").date() == datetime.strptime(date, "%Y-%m-%d").date():
                games.append({
                    "GAME_ID": game.get("gid"),
                    "TEAM_ID": int(game["h"]["tid"]),
                    "OPPONENT_TEAM_ID": int(game["v"]["tid"]),
                    "date": date,
                })
                games.append({
                    "GAME_ID": game.get("gid"),
                    "TEAM_ID": int(game["v"]["tid"]),
                    "OPPONENT_TEAM_ID": int(game["h"]["tid"]),
                    "date": date,
                })
    return pd.DataFrame(games, columns=EXPECTED_GAME_COLS)

def fetch_boxscores(game_ids: List[str], use_cache=True) -> pd.DataFrame:
    """Fetch box scores for a list of games from JSON API."""
    all_rows = []
    for gid in game_ids:
        cache_file = f"boxscore_{gid}.json"
        data = _load_cache(cache_file) if use_cache else None

        if data is None:
            try:
                url = f"{BASE_URL}/games/{gid}_boxscore.json"
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                data = resp.json()
                _cache_response(cache_file, data)
            except Exception as e:
                logging.error(f"Failed to fetch boxscore for {gid}: {e}")
                continue

        game_data = data.get("g", {}).get("pd", {})
        for team_side in ["h", "v"]:
            team = game_data.get(team_side)
            if not team:
                continue
            team_id = int(team.get("tid", 0))
            for player in team.get("pstsg", []):
                all_rows.append({
                    "GAME_ID": gid,
                    "PLAYER_ID": int(player.get("pid", 0)),
                    "TEAM_ID": team_id,
                    "PTS": int(player.get("pts", 0)),
                    "REB": int(player.get("reb", 0)),
                    "AST": int(player.get("ast", 0)),
                    "MIN": player.get("min", "0:00"),
                })
    return pd.DataFrame(all_rows, columns=EXPECTED_BOX_COLS)

# -----------------------------
# nba_api.stats ENDPOINTS
# -----------------------------
def fetch_season_games(season: str) -> pd.DataFrame:
    """Fetch all games for a given season using nba_api.stats."""
    logging.info(f"Fetching games for season {season}...")
    try:
        gamefinder = leaguegamefinder.LeagueGameFinder(season_nullable=season)
        df = gamefinder.get_data_frames()[0]
        df = df[["GAME_DATE", "TEAM_NAME", "MATCHUP"]]
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        return df
    except Exception as e:
        logging.error(f"Failed to fetch season {season}: {e}")
        return pd.DataFrame(columns=["GAME_DATE", "TEAM_NAME", "MATCHUP"])

def fetch_today_games() -> pd.DataFrame:
    """Fetch today's NBA games using nba_api.stats."""
    today_str = datetime.now().strftime("%Y-%m-%d")
    try:
        current_season = SEASONS[-1]
        gamefinder = leaguegamefinder.LeagueGameFinder(season_nullable=current_season)
        games = gamefinder.get_data_frames()[0]
    except Exception as e:
        logging.error(f"Failed to fetch today's games: {e}")
        return pd.DataFrame(columns=["GAME_DATE", "TEAM_NAME", "MATCHUP", "home_team", "away_team"])

    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])
    today_games = games[games["GAME_DATE"] == pd.to_datetime(today_str)]
    if today_games.empty:
        return pd.DataFrame(columns=["GAME_DATE", "TEAM_NAME", "MATCHUP", "home_team", "away_team"])

    today_games["home_team"] = today_games["MATCHUP"].apply(
        lambda x: x.split(" vs. ")[0] if "vs." in x else x.split(" @ ")[1]
    )
    today_games["away_team"] = today_games["MATCHUP"].apply(
        lambda x: x.split(" vs. ")[1] if "vs." in x else x.split(" @ ")[0]
    )
    return today_games.dropna(subset=["home_team", "away_team"])

def update_historical_games(existing_df: pd.DataFrame) -> pd.DataFrame:
    """Fetch and combine historical games for all seasons using nba_api.stats."""
    all_new_games = []
    for season in SEASONS:
        new_games = fetch_season_games(season)
        if not new_games.empty:
            all_new_games.append(new_games)

    if all_new_games:
        combined_df = pd.concat([existing_df] + all_new_games, ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=["GAME_DATE", "TEAM_NAME", "MATCHUP"])
    else:
        combined_df = existing_df

    logging.info(f"Historical games updated. Total games: {len(combined_df)}")
    return combined_df
