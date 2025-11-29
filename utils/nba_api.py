import requests
import pandas as pd
import logging

BASE_URL = "https://www.balldontlie.io/api/v1/games"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def fetch_nba_games(season):
    logging.info(f"🚀 Fetching NBA games for {season} season")
    games = []
    page = 1
    while True:
        resp = requests.get(BASE_URL, params={"seasons[]": season, "per_page": 100, "page": page})
        resp.raise_for_status()
        data = resp.json()
        if not data["data"]:
            break
        games.extend(data["data"])
        if page >= data["meta"]["total_pages"]:
            break
        page += 1
    df = pd.DataFrame(games)
    if df.empty:
        logging.warning("❌ No games fetched")
        return pd.DataFrame()
    # Flatten team info
    df["HOME_TEAM"] = df["home_team"].apply(lambda x: x["full_name"])
    df["VISITOR_TEAM"] = df["visitor_team"].apply(lambda x: x["full_name"])
    df["HOME_SCORE"] = df["home_team_score"]
    df["VISITOR_SCORE"] = df["visitor_team_score"]
    df = df[["id", "date", "HOME_TEAM", "VISITOR_TEAM", "HOME_SCORE", "VISITOR_SCORE"]]
    return df
