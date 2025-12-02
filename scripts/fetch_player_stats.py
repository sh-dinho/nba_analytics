import os
import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def main(season="2024-25"):
    """Fetch latest NBA player stats and save locally."""
    print(f"Fetching player stats for season {season}...")
    try:
        stats = leaguedashplayerstats.LeagueDashPlayerStats(season=season)
        df = stats.get_data_frames()[0]
        df.to_csv(f"{DATA_DIR}/player_stats.csv", index=False)
        print(f"Player stats saved to {DATA_DIR}/player_stats.csv")
    except Exception as e:
        print(f"Failed to fetch player stats: {e}")
        raise
