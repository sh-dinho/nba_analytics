import os
import sqlite3
import yaml
import logging
import requests
import pandas as pd
from datetime import datetime
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ROOT = os.path.dirname(os.path.dirname(__file__))
CONFIG = yaml.safe_load(open(os.path.join(ROOT, "config.yaml")))

DB_PATH = CONFIG["database"]["path"]
SEASONS = CONFIG["data"]["seasons_to_fetch"]


# -----------------------------------------------------
# ESPN FREE API — Primary Data Source (No Key Needed)
# -----------------------------------------------------
def fetch_espn_season(season: int) -> pd.DataFrame:
    """Fetch all NBA games for a season via ESPN's public JSON API."""
    all_games = []

    months = [
        "october", "november", "december",
        "january", "february", "march",
        "april", "may", "june"
    ]

    for month in months:
        url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={season}{month}"
        logging.info(f"Fetching ESPN data: {url}")

        try:
            data = requests.get(url, timeout=10).json()

            if "events" not in data:
                continue

            for e in data["events"]:
                comp = e["competitions"][0]
                teams = comp["competitors"]

                game = {
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "Date": e.get("date", "")[:10],
                    "Visitor": teams[1]["team"]["displayName"],
                    "Visitor_PTS": teams[1].get("score", None),
                    "Home": teams[0]["team"]["displayName"],
                    "Home_PTS": teams[0].get("score", None),
                    "Overtime": "OT" if comp.get("status", {}).get("period", 4) > 4 else "",
                    "Notes": ""
                }
                all_games.append(game)

            time.sleep(0.3)

        except Exception as e:
            logging.error(f"Failed ESPN fetch for month {month}: {e}")

    return pd.DataFrame(all_games)



# -----------------------------------------------------
# Optional Backup — Basketball Reference (Spoof headers)
# -----------------------------------------------------
def fetch_bbr_season(season: int) -> pd.DataFrame:
    """Fallback scraper for BBR."""
    import pandas as pd

    url = f"https://www.basketball-reference.com/leagues/NBA_{season}_games.html"
    logging.info(f"Fetching backup BBR data from {url}")

    try:
        df = pd.read_html(url, header=0)[0]
        df = df.dropna(subset=["Visitor/Neutral"])

        df = df.rename(columns={
            "Visitor/Neutral": "Visitor",
            "PTS": "Visitor_PTS",
            "Home/Neutral": "Home",
            "PTS.1": "Home_PTS"
        })

        df["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        df["Overtime"] = df["Unnamed: 7"].fillna("")
        df["Notes"] = df["Unnamed: 8"].fillna("")

        df = df[[
            "Timestamp", "Date", "Visitor", "Visitor_PTS",
            "Home", "Home_PTS", "Overtime", "Notes"
        ]]

        return df

    except Exception as e:
        logging.error(f"BBR failed: {e}")
        return pd.DataFrame([])



# -----------------------------------------------------
# Store games in SQLite
# -----------------------------------------------------
def store_games(df: pd.DataFrame):
    if df.empty:
        logging.error("❌ No game data to store.")
        return

    with sqlite3.connect(DB_PATH) as con:
        df.to_sql("nba_games", con, if_exists="append", index=False)

    logging.info(f"✔ Stored {len(df)} games.")



# -----------------------------------------------------
# Main runner
# -----------------------------------------------------
def fetch_all():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    all_data = pd.DataFrame()

    for season in SEASONS:
        logging.info(f"🚀 Fetching season: {season}")

        df = fetch_espn_season(season)

        if df.empty:
            logging.warning("ESPN returned nothing. Trying BBR...")
            df = fetch_bbr_season(season)

        if df.empty:
            logging.error(f"❌ Could not fetch season {season} from any source.")
            continue

        all_data = pd.concat([all_data, df], ignore_index=True)

    if all_data.empty:
        logging.error("❌ No game data fetched for any season.")
        return

    store_games(all_data)



# -----------------------------------------------------
# Direct execution
# -----------------------------------------------------
if __name__ == "__main__":
    logging.info("🏀 Fetching NBA historical games...")
    fetch_all()
    logging.info("🎉 Game fetch complete.")
