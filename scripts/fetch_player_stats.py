# File: scripts/fetch_player_stats.py

import os
import time
import pandas as pd
import logging
import datetime
import requests
import numpy as np
import argparse

# Directories and cache file
DATA_DIR = "data"
RESULTS_DIR = "results"
CACHE_FILE = f"{DATA_DIR}/player_stats.csv"
SUMMARY_LOG = f"{RESULTS_DIR}/fetch_summary.csv"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# NBA.com headers
NBA_HEADERS = {
    "Host": "stats.nba.com",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0)",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://www.nba.com",
    "Referer": "https://www.nba.com/",
    "Connection": "keep-alive",
}

LEAGUE_DASH_URL = (
    "https://stats.nba.com/stats/leaguedashplayerstats?"
    "Season={season}&SeasonType=Regular+Season&MeasureType=Base&PerMode=PerGame"
)

def generate_synthetic_stats(n_players=450):
    """Generates synthetic player stats for testing/fallback."""
    logging.info("Generating synthetic player stats...")
    data = {
        "PLAYER_NAME": [f"Player_{i}" for i in range(n_players)],
        "TEAM_ABBREVIATION": np.random.choice(
            ["LAL", "BOS", "GSW", "MIL", "DEN", "PHX", "BKN", "DAL", "PHI", "TOR"],
            n_players
        ),
        "PTS": np.random.uniform(5, 30, n_players),
        "AST": np.random.uniform(1, 10, n_players),
        "REB": np.random.uniform(2, 15, n_players),
        "STL": np.random.uniform(0, 2, n_players),
        "BLK": np.random.uniform(0, 2, n_players),
    }
    return pd.DataFrame(data)

def fetch_stats_direct(season: str, timeout=30):
    """Fetch stats from NBA.com API."""
    url = LEAGUE_DASH_URL.format(season=season)
    r = requests.get(url, headers=NBA_HEADERS, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    headers = data["resultSets"][0]["headers"]
    rows = data["resultSets"][0]["rowSet"]
    return pd.DataFrame(rows, columns=headers)

def fetch_stats_bball_ref(season_year=2025):
    """Fetch per-game player stats from Basketball Reference."""
    url = f"https://www.basketball-reference.com/leagues/NBA_{season_year}_per_game.html"
    tables = pd.read_html(url)
    df = tables[0]
    df = df[df["Player"] != "Player"]  # drop repeated header rows
    df.rename(columns={
        "Player": "PLAYER_NAME",
        "Tm": "TEAM_ABBREVIATION",
        "PTS": "PTS",
        "AST": "AST",
        "TRB": "REB",
        "STL": "STL",
        "BLK": "BLK"
    }, inplace=True)
    return df[["PLAYER_NAME", "TEAM_ABBREVIATION", "PTS", "AST", "REB", "STL", "BLK"]]

def main(season="2024-25", retries=4, base_delay=5, max_total_wait=60,
          use_synthetic=False, source="nba"):
    logging.info(f"Fetching player stats for season {season} using source={source}...")

    df, source_used = None, None

    if use_synthetic:
        df = generate_synthetic_stats()
        source_used = "synthetic_forced"

    elif source == "bball_ref":
        season_year = int(season.split("-")[0]) + 1  # "2024-25" -> 2025
        try:
            df = fetch_stats_bball_ref(season_year)
            source_used = "basketball_reference"
            logging.info(f"✅ Player stats fetched from Basketball Reference")
        except Exception as e:
            logging.warning(f"Basketball Reference fetch failed: {e}. Falling back to NBA API.")
            source = "nba"


    if df is None and source == "nba":  # default: nba.com (or fallback from bball_ref)
        total_wait, attempts = 0, 0
        for attempt in range(1, retries + 1):
            attempts = attempt
            try:
                df = fetch_stats_direct(season, timeout=30)
                source_used = "nba_api"
                logging.info(f"✅ Player stats fetched from NBA.com")
                break
            except Exception as e:
                logging.warning(f"Attempt {attempt} failed: {e}")
                if attempt < retries:
                    delay = min(base_delay * (2 ** (attempt - 1)), max_total_wait - total_wait)
                    if delay <= 0:
                        break
                    total_wait += delay
                    logging.info(f"Retrying in {delay}s (total waited: {total_wait}s)...")
                    time.sleep(delay)
                else:
                    logging.error("❌ All retries failed.")
                    if os.path.exists(CACHE_FILE):
                        logging.info(f"Loading cached data from {CACHE_FILE}")
                        df = pd.read_csv(CACHE_FILE)
                        source_used = "cache"
                    else:
                        logging.warning("⚠️ No cached data available. Falling back to synthetic.")
                        df = generate_synthetic_stats()
                        source_used = "synthetic_fallback"

    if df is None:
        raise RuntimeError("Fetch failed with no data source available.")

    df.to_csv(CACHE_FILE, index=False)

    # Log summary
    run_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary_entry = pd.DataFrame([{
        "timestamp": run_time,
        "season": season,
        "source": source_used,
        "rows_fetched": len(df)
    }])
    summary_entry.to_csv(
        SUMMARY_LOG,
        mode="a" if os.path.exists(SUMMARY_LOG) else "w",
        index=False,
        header=not os.path.exists(SUMMARY_LOG),
    )
    logging.info(f"📁 Fetch summary appended to {SUMMARY_LOG}")
    logging.info(f"✅ Player stats saved to {CACHE_FILE}")
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch NBA player stats")
    parser.add_argument("--season", type=str, default="2024-25", help="Season string, e.g. 2024-25")
    parser.add_argument("--source", type=str, choices=["nba", "bball_ref", "synthetic"], default="nba",
                         help="Data source: nba (default), bball_ref, synthetic")
    parser.add_argument("--use_synthetic", action="store_true", help="Force synthetic stats")
    args = parser.parse_args()

    main(season=args.season, use_synthetic=args.use_synthetic, source=args.source)