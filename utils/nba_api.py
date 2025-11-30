import pandas as pd
import requests

def fetch_nba_games(season: int):
    """Fetch NBA game results from free API (example: balldontlie.io)"""
    games = []
    page = 1
    while True:
        url = f"https://www.balldontlie.io/api/v1/games?seasons[]={season}&per_page=100&page={page}"
        resp = requests.get(url)
        if resp.status_code != 200:
            break
        data = resp.json()
        if not data["data"]:
            break
        games.extend(data["data"])
        page += 1
    if not games:
        raise ValueError("No NBA game data fetched.")
    df = pd.DataFrame(games)
    return df
