import requests
import sqlite3
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

DB_PATH = "nba_analytics.db"
SEASON = 2023
PER_PAGE = 100

def fetch_games(season):
    logging.info(f"🚀 Fetching NBA games for season {season}")
    games = []
    page = 1
    while True:
        url = f"https://www.balldontlie.io/api/v1/games?seasons[]={season}&per_page={PER_PAGE}&page={page}"
        resp = requests.get(url)
        if resp.status_code != 200:
            logging.warning(f"HTTP error on page {page}: {resp.status_code}")
            break
        data = resp.json()
        games.extend(data["data"])
        if page >= data["meta"]["total_pages"]:
            break
        page += 1
    if not games:
        logging.error("⚠ No games fetched")
        return pd.DataFrame()
    return pd.DataFrame(games)

def store_games(df):
    if df.empty:
        logging.error("❌ No NBA game data found. Cannot proceed.")
        return
    df_games = df[["id", "home_team_id", "visitor_team_id", "home_team_score", "visitor_team_score", "season", "date"]]
    df_games.columns = ["game_id", "home_team", "away_team", "home_team_score", "away_team_score", "season", "date"]
    with sqlite3.connect(DB_PATH) as con:
        df_games.to_sql("nba_games", con, if_exists="replace", index=False)
    logging.info(f"✔ Stored {len(df_games)} games in database.")

if __name__ == "__main__":
    df_games = fetch_games(SEASON)
    store_games(df_games)
