# ============================================================
# Path: src/api/nba_api.py
# Purpose: Fetch NBA games and boxscores from API
# Version: 1.0 (2022+, structured for pipeline)
# ============================================================

import requests
import pandas as pd
from datetime import datetime

BASE_URL = "https://data.nba.com/data/v2015/json/mobile_teams/nba"


def fetch_games(date: str) -> pd.DataFrame:
    """
    Fetch NBA games for a given date.
    date: "YYYY-MM-DD"
    Returns a DataFrame with GAME_ID, TEAM_ID, OPPONENT_TEAM_ID, date
    """
    try:
        y, m, d = date.split("-")
        url = f"{BASE_URL}/{y}/league/00_full_schedule.json"
        resp = requests.get(url)
        resp.raise_for_status()
        data = resp.json()

        # Flatten schedule to get games on the requested date
        games = []
        for month in data['lscd']:
            for game in month['mscd']['g']:
                game_date = game['gdte']
                if game_date == date:
                    games.append({
                        "GAME_ID": game['gid'],
                        "TEAM_ID": int(game['h']['tid']),
                        "OPPONENT_TEAM_ID": int(game['v']['tid']),
                        "date": date
                    })
                    games.append({
                        "GAME_ID": game['gid'],
                        "TEAM_ID": int(game['v']['tid']),
                        "OPPONENT_TEAM_ID": int(game['h']['tid']),
                        "date": date
                    })
        return pd.DataFrame(games)
    except Exception as e:
        print(f"Failed to fetch games: {e}")
        return pd.DataFrame()


def fetch_boxscores(game_ids: list[str]) -> pd.DataFrame:
    """
    Fetch box scores for a list of games.
    Returns DataFrame with PLAYER_ID, TEAM_ID, PTS, REB, AST, etc.
    """
    all_rows = []
    for gid in game_ids:
        try:
            url = f"{BASE_URL}/games/{gid}_boxscore.json"
            resp = requests.get(url)
            resp.raise_for_status()
            data = resp.json()

            for team_side in ['h', 'v']:
                team = data['g']['pd'][team_side]
                team_id = int(team['tid'])
                for player in team['pstsg']:
                    all_rows.append({
                        "GAME_ID": gid,
                        "PLAYER_ID": int(player['pid']),
                        "TEAM_ID": team_id,
                        "PTS": int(player.get('pts', 0)),
                        "REB": int(player.get('reb', 0)),
                        "AST": int(player.get('ast', 0)),
                        "MIN": player.get('min', "0:00")
                    })
        except Exception as e:
            print(f"Failed to fetch boxscore for {gid}: {e}")
            continue
    return pd.DataFrame(all_rows)
