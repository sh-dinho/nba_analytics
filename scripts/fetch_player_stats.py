import os
import time
import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats
from nba_api.library.http import NBAStatsHTTP

DATA_DIR = "data"
CACHE_FILE = f"{DATA_DIR}/player_stats.csv"
os.makedirs(DATA_DIR, exist_ok=True)

# Make requests look more like a browser
NBAStatsHTTP.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
})
NBAStatsHTTP.timeout = 60  # increase timeout

def main(season="2024-25", retries=3, delay=10):
    """Fetch latest NBA player stats and save locally. Falls back to cached data if API fails."""
    print(f"Fetching player stats for season {season}...")
    for attempt in range(1, retries + 1):
        try:
            stats = leaguedashplayerstats.LeagueDashPlayerStats(season=season)
            df = stats.get_data_frames()[0]
            df.to_csv(CACHE_FILE, index=False)
            print(f"Player stats saved to {CACHE_FILE}")
            return df
        except Exception as e:
            print(f"Attempt {attempt} failed: {e}")
            if attempt < retries:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("All retries failed.")
                if os.path.exists(CACHE_FILE):
                    print(f"Loading cached data from {CACHE_FILE}")
                    return pd.read_csv(CACHE_FILE)
                else:
                    raise RuntimeError("No cached data available and API fetch failed.")