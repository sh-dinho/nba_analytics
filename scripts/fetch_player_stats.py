# File: scripts/fetch_player_stats.py

import os
import pandas as pd
import logging
import datetime
import argparse

DATA_DIR = "data"
RESULTS_DIR = "results"
CACHE_FILE = os.path.join(DATA_DIR, "player_stats.csv")
SUMMARY_LOG = os.path.join(RESULTS_DIR, "fetch_summary.csv")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fetch_player_stats")

def fetch_stats_bball_ref(season_year=2025):
    url = f"https://www.basketball-reference.com/leagues/NBA_{season_year}_per_game.html"
    tables = pd.read_html(url)
    df = tables[0]
    df = df[df["Player"] != "Player"]
    df.rename(columns={"Player": "PLAYER_NAME", "Tm": "TEAM_ABBREVIATION", "TRB": "REB"}, inplace=True)
    cols = ["PLAYER_NAME", "TEAM_ABBREVIATION", "PTS", "AST", "REB", "STL", "BLK"]
    return df[cols]

def main(season="2024-25", force_refresh=False):
    logger.info(f"Fetching player stats for season {season}...")
    if os.path.exists(CACHE_FILE) and not force_refresh:
        logger.info(f"Using cached stats from {CACHE_FILE}")
        df = pd.read_csv(CACHE_FILE)
        source_used = "cache"
    else:
        season_year = int(season.split("-")[0]) + 1
        df = fetch_stats_bball_ref(season_year)
        df.to_csv(CACHE_FILE, index=False)
        source_used = "basketball_reference"
        logger.info(f"Scraped and saved stats to {CACHE_FILE}")

    run_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary_entry = pd.DataFrame([{
        "timestamp": run_time, "season": season, "source": source_used, "rows_fetched": len(df)
    }])
    summary_entry.to_csv(
        SUMMARY_LOG,
        mode="a" if os.path.exists(SUMMARY_LOG) else "w",
        index=False,
        header=not os.path.exists(SUMMARY_LOG),
    )
    logger.info(f"Fetch summary appended to {SUMMARY_LOG}")
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch NBA player stats (cache-aware)")
    parser.add_argument("--season", type=str, default="2024-25")
    parser.add_argument("--force_refresh", action="store_true")
    args = parser.parse_args()
    main(season=args.season, force_refresh=args.force_refresh)