import requests
import logging
import sqlite3
import time
import yaml
import os
import pandas as pd

from utils.db_init import init_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# -----------------------------
# Load config
# -----------------------------
CONFIG_PATH = "config.yaml"
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Missing config.yaml at {CONFIG_PATH}")

CONFIG = yaml.safe_load(open(CONFIG_PATH))
DB_PATH = CONFIG["database"]["path"]

# -----------------------------
# Constants
# -----------------------------
API_URL = "https://api.balldontlie.io/v1/games"
API_KEY = CONFIG["api"]["balldontlie_key"]   # FREE key (user must add)

HEADERS = {"Authorization": API_KEY} if API_KEY else {}


# -----------------------------
# Fetch season games
# -----------------------------
def fetch_season_games(season: int, retries: int = 3):
    logging.info(f"📅 Fetching NBA games for season {season}...")

    games = []
    page = 1

    while True:
        url = f"{API_URL}?seasons[]={season}&per_page=100&page={page}"
        logging.info(f"Fetching page {page}...")

        attempt = 0
        while attempt < retries:
            try:
                r = requests.get(url, headers=HEADERS, timeout=10)
                r.raise_for_status()
                data = r.json()

                if "data" not in data:
                    raise ValueError("Invalid response format")

                games.extend(data["data"])

                if page >= data["meta"]["total_pages"]:
                    logging.info(f"✔ Completed season {season}. {len(games)} games fetched.")
                    return games

                page += 1
                break

            except Exception as e:
                attempt += 1
                logging.warning(f"Retry {attempt}/{retries} failed: {e}")
                time.sleep(2)

        if attempt == retries:
            logging.error(f"❌ Failed to fetch page {page} after {retries} attempts.")
            return games


# -----------------------------
# Store into database
# -----------------------------
def store_games(games: list, season: int):
    if not games:
        logging.error(f"❌ No games to store for season {season}")
        return

    logging.info(f"🗂 Storing {len(games)} games for season {season}...")

    rows = []
    for g in games:
        rows.append({
            "GameID": g["id"],
            "Date": g["date"].split("T")[0],
            "Visitor": g["visitor_team"]["abbreviation"],
            "VisitorPts": g["visitor_team_score"],
            "Home": g["home_team"]["abbreviation"],
            "HomePts": g["home_team_score"],
            "Winner": g["home_team"]["abbreviation"] if g["home_team_score"] > g["visitor_team_score"]
                      else g["visitor_team"]["abbreviation"],
            "Season": season
        })

    df = pd.DataFrame(rows)

    conn = sqlite3.connect(DB_PATH)
    df.to_sql("nba_games", conn, if_exists="append", index=False)
    conn.close()

    logging.info(f"✅ Stored {len(df)} rows for season {season}")


# -----------------------------
# MAIN EXECUTION
# -----------------------------
if __name__ == "__main__":
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    # Initialize DB if not created
    init_db(DB_PATH)

    start = CONFIG["fetcher"]["start_season"]
    end = CONFIG["fetcher"]["end_season"]

    logging.info(f"🚀 Fetching games from {start} to {end}")

    for season in range(start, end + 1):
        games = fetch_season_games(season)
        store_games(games, season)

    logging.info("🎉 DONE — All seasons fetched successfully")
