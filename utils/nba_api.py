import requests
import pandas as pd
from datetime import datetime

def fetch_nba_games(season: int):
    """Fetch NBA games from the free API."""
    url = f"https://www.balldontlie.io/api/v1/games?seasons[]={season}&per_page=100"
    all_games = []
    page = 1

    while True:
        resp = requests.get(url + f"&page={page}")
        resp.raise_for_status()
        data = resp.json()
        all_games.extend(data["data"])
        if data["meta"]["next_page"] is None:
            break
        page += 1

    if not all_games:
        raise ValueError(f"No games found for season {season}")

    df = pd.DataFrame(all_games)
    return df
