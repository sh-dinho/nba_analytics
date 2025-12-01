# nba_analytics_core/data.py
import logging
from typing import List, Dict, Any
from config import TEAM_MAP

def _normalize_team(name: str) -> str:
    return TEAM_MAP.get(name, name)

def fetch_historical_games(season: int | None = None) -> List[Dict[str, Any]]:
    logging.info(f"Fetching historical games{' for season ' + str(season) if season else ''}...")
    # Placeholder sample; replace with real source
    data = [
        {"game_id": "2025-001", "season": 2025, "date": "2025-10-20", "home_team": _normalize_team("LAL"), "away_team": _normalize_team("BOS"), "home_score": 110, "away_score": 108},
        {"game_id": "2025-002", "season": 2025, "date": "2025-10-21", "home_team": _normalize_team("NYK"), "away_team": _normalize_team("BOS"), "home_score": 95, "away_score": 100},
    ]
    if season:
        data = [d for d in data if d["season"] == season]
    logging.info(f"✔ Fetched {len(data)} games")
    return data

def engineer_features(df):
    # df: pandas DataFrame with scores
    import pandas as pd
    result = df.copy()
    result["home_win"] = (result["home_score"] > result["away_score"]).where(
        ~(result["home_score"].isna() | result["away_score"].isna()), other=pd.NA
    )
    result["total_points"] = (result["home_score"] + result["away_score"]).where(
        ~(result["home_score"].isna() | result["away_score"].isna()), other=pd.NA
    )
    return result