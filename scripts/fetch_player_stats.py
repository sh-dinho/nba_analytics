# File: scripts/fetch_player_stats.py

import os
import time
import pandas as pd
import logging
import datetime
import requests
import numpy as np # <-- NEW IMPORT

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

# Mandatory headers — nba.com blocks requests without these
NBA_HEADERS = {
    "Host": "stats.nba.com",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0)",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://www.nba.com",
    "Referer": "https://www.nba.com/",
    "Connection": "keep-alive",
}

# Stable direct endpoint avoids nba_api issues
LEAGUE_DASH_URL = (
    "https://stats.nba.com/stats/leaguedashplayerstats?"
    "Season={season}&SeasonType=Regular+Season&MeasureType=Base&PerMode=PerGame"
)

def generate_synthetic_stats(n_players=450): # <-- NEW FUNCTION
    """Generates synthetic player stats for testing/fallback."""
    logging.info("Generating synthetic player stats...")
    data = {
        "PLAYER_NAME": [f"Player_{i}" for i in range(n_players)],
        "TEAM_ABBREVIATION": np.random.choice(["LAL", "BOS", "GSW", "MIL", "DEN", "PHX", "BKN", "DAL", "PHI", "TOR"], n_players),
        "PTS": np.random.uniform(5, 30, n_players),
        "AST": np.random.uniform(1, 10, n_players),
        "REB": np.random.uniform(2, 15, n_players),
        "STL": np.random.uniform(0, 2, n_players),
        "BLK": np.random.uniform(0, 2, n_players),
    }
    df = pd.DataFrame(data)
    # Ensure required columns for downstream tasks are present
    required_cols = ["PLAYER_NAME", "TEAM_ABBREVIATION", "PTS", "AST", "REB", "STL", "BLK"]
    return df[required_cols]

def fetch_stats_direct(season: str, timeout=30): # <-- TIMEOUT INCREASED (15s to 30s)
    url = LEAGUE_DASH_URL.format(season=season)
    r = requests.get(url, headers=NBA_HEADERS, timeout=timeout)
    r.raise_for_status()

    data = r.json()
    headers = data["resultSets"][0]["headers"]
    rows = data["resultSets"][0]["rowSet"]
    return pd.DataFrame(rows, columns=headers)

def main(season="2024-25", retries=4, base_delay=5, max_total_wait=60, use_synthetic=False): # <-- ADD use_synthetic
    logging.info(f"Fetching player stats for season {season}...")

    # --- Synthetic Data Logic (NEW) ---
    if use_synthetic:
        df = generate_synthetic_stats()
        df.to_csv(CACHE_FILE, index=False)
        source = "Synthetic"
        
        # Log successful synthetic run before returning
        run_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        summary_entry = pd.DataFrame([{
            "timestamp": run_time,
            "season": season,
            "source": source,
            "attempts": 1,
            "total_wait": 0,
            "rows_fetched": len(df)
        }])
        summary_entry.to_csv(
            SUMMARY_LOG,
            mode="a" if os.path.exists(SUMMARY_LOG) else "w",
            index=False,
            header=not os.path.exists(SUMMARY_LOG),
        )
        logging.info(f"📁 Fetch summary appended to {SUMMARY_LOG} (Synthetic)")
        logging.info(f"✅ Synthetic player stats saved to {CACHE_FILE}")
        return df
    # ----------------------------------


    total_wait = 0
    attempts = 0
    source = None
    df = None

    for attempt in range(1, retries + 1):
        attempts = attempt

        try:
            df = fetch_stats_direct(season, timeout=30) # <-- TIMEOUT INCREASED (12s to 30s)
            df.to_csv(CACHE_FILE, index=False)
            source = "API"
            logging.info(f"✅ Player stats saved to {CACHE_FILE}")
            break

        except Exception as e:
            logging.warning(f"Attempt {attempt} failed: {e}")

            if attempt < retries:
                delay = min(base_delay * (2 ** (attempt - 1)), max_total_wait - total_wait)
                if delay <= 0:
                    logging.error("❌ Reached max total wait, aborting retries.")
                    break

                total_wait += delay
                logging.info(f"Retrying in {delay}s (total waited: {total_wait}s)...")
                time.sleep(delay)

            else:
                logging.error("❌ All retries failed.")
                if os.path.exists(CACHE_FILE):
                    logging.info(f"Loading cached data from {CACHE_FILE}")
                    df = pd.read_csv(CACHE_FILE)
                    source = "cache"
                else:
                    raise RuntimeError(
                        "No cached data available and API fetch failed."
                    )

    # Summary log
    if df is None:
        raise RuntimeError("Fetch failed with no data source available.")

    run_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    summary_entry = pd.DataFrame([{
        "timestamp": run_time,
        "season": season,
        "source": source,
        "attempts": attempts,
        "total_wait": total_wait,
        "rows_fetched": len(df)
    }])

    summary_entry.to_csv(
        SUMMARY_LOG,
        mode="a" if os.path.exists(SUMMARY_LOG) else "w",
        index=False,
        header=not os.path.exists(SUMMARY_LOG),
    )

    logging.info(f"📁 Fetch summary appended to {SUMMARY_LOG}")
    return df


if __name__ == "__main__":
    main()