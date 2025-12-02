# File: scripts/fetch_player_stats.py

import os
import time
import pandas as pd
import logging
import datetime
from nba_api.stats.endpoints import leaguedashplayerstats

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

# Custom headers to mimic a browser
CUSTOM_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Referer": "https://www.nba.com/",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "application/json, text/plain, */*",
    "Origin": "https://www.nba.com",
    "Connection": "keep-alive"
}

def main(season="2024-25", retries=3, base_delay=10, max_total_wait=120):
    """
    Fetch latest NBA player stats and save locally.
    Falls back to cached data if API fails.
    Uses exponential backoff for retries with a maximum total wait cap.
    Appends a summary log to results/fetch_summary.csv.
    """
    logging.info(f"Fetching player stats for season {season}...")

    total_wait = 0
    source = None
    attempts = 0
    df = None

    for attempt in range(1, retries + 1):
        attempts = attempt
        try:
            stats = leaguedashplayerstats.LeagueDashPlayerStats(
                season=season,
                timeout=30,
                headers=CUSTOM_HEADERS
            )
            df = stats.get_data_frames()[0]
            df.to_csv(CACHE_FILE, index=False)

            source = "API"
            logging.info(f"✅ Player stats saved to {CACHE_FILE}")
            break

        except Exception as e:
            logging.warning(f"Attempt {attempt} failed: {e}")
            if attempt < retries:
                # Exponential backoff: delay doubles each attempt
                delay = base_delay * (2 ** (attempt - 1))
                # Enforce maximum total wait cap
                if total_wait + delay > max_total_wait:
                    delay = max_total_wait - total_wait
                if delay <= 0:
                    logging.error("❌ Max total wait reached, stopping retries.")
                    break

                total_wait += delay
                logging.info(f"Retrying in {delay} seconds (total waited: {total_wait}s)...")
                time.sleep(delay)
            else:
                logging.error("❌ All retries failed.")
                if os.path.exists(CACHE_FILE):
                    logging.info(f"Loading cached data from {CACHE_FILE}")
                    df = pd.read_csv(CACHE_FILE)
                    source = "cache"
                    break
                else:
                    raise RuntimeError("No cached data available and API fetch failed.") from e

    # Final summary log
    if source and df is not None:
        run_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logging.info(
            f"📊 Fetch completed: source={source}, attempts={attempts}, total_wait={total_wait}s"
        )

        # Append summary to rolling log
        summary_entry = pd.DataFrame([{
            "timestamp": run_time,
            "season": season,
            "source": source,
            "attempts": attempts,
            "total_wait": total_wait,
            "rows_fetched": len(df)
        }])

        if os.path.exists(SUMMARY_LOG):
            summary_entry.to_csv(SUMMARY_LOG, mode="a", header=False, index=False)
        else:
            summary_entry.to_csv(SUMMARY_LOG, index=False)

        logging.info(f"📁 Fetch summary appended to {SUMMARY_LOG}")
        return df
    else:
        raise RuntimeError("Fetch failed with no data source available.")

if __name__ == "__main__":
    main()