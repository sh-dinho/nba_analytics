import requests
import pandas as pd
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

API_BASE = "https://www.balldontlie.io/api/v1"

def fetch_nba_games(season: int = 2025, per_page: int = 100) -> pd.DataFrame:
    """
    Fetch NBA game data from balldontlie.io for a given season.
    Returns a pandas DataFrame with columns:
    - date, home_team, visitor_team, home_score, visitor_score, winner
    """
    logging.info(f"🚀 Fetching NBA games for season {season}")
    all_games = []
    page = 1

    while True:
        url = f"{API_BASE}/games"
        params = {
            "seasons[]": season,
            "per_page": per_page,
            "page": page
        }
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if "data" not in data or not data["data"]:
                break

            for g in data["data"]:
                home_score = g["home_team_score"]
                visitor_score = g["visitor_team_score"]
                winner = g["home_team"]["full_name"] if home_score > visitor_score else g["visitor_team"]["full_name"]
                all_games.append({
                    "date": g["date"],
                    "home_team": g["home_team"]["full_name"],
                    "visitor_team": g["visitor_team"]["full_name"],
                    "home_score": home_score,
                    "visitor_score": visitor_score,
                    "winner": winner
                })

            if page >= data["meta"]["total_pages"]:
                break
            page += 1

        except requests.HTTPError as e:
            logging.warning(f"HTTP error on page {page}: {e}")
            break
        except Exception as e:
            logging.error(f"Failed to fetch page {page}: {e}")
            break

    df = pd.DataFrame(all_games)
    if df.empty:
        logging.warning("⚠ No games fetched")
    else:
        logging.info(f"✔ Fetched {len(df)} games")
        df["date"] = pd.to_datetime(df["date"])
    return df
