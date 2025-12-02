# File: scripts/fetch_player_stats.py

import os
import time
import pandas as pd
import logging
import datetime
import requests

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

def fetch_stats_direct(season: str, timeout=15):
    url = LEAGUE_DASH_URL.format(season=season)
    r = requests.get(url, headers=NBA_HEADERS, timeout=timeout)
    r.raise_for_status()

    data = r.json()
    headers = data["resultSets"][0]["headers"]
    rows = data["resultSets"][0]["rowSet"]
    return pd.DataFrame(rows, columns=headers)

def main(season="2024-25", retries=4, base_delay=5, max_total_wait=60):
    logging.info(f"Fetching player stats for season {season}...")

    total_wait = 0
    attempts = 0
    source = None
    df = None

    for attempt in range(1, retries + 1):
        attempts = attempt

        try:
            df = fetch_stats_direct(season, timeout=12)
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
