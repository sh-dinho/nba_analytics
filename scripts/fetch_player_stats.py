# scripts/fetch_player_stats.py
import pandas as pd
import os
from scripts.utils import setup_logger

logger = setup_logger("fetch_player_stats")

def main(season="2024-25", resume=True):
    """
    Fetch or update player stats for the given season.
    For now, this is a placeholder to generate dummy data.
    """
    os.makedirs("data", exist_ok=True)
    stats_file = f"data/player_stats_{season}.csv"

    if resume and os.path.exists(stats_file):
        logger.info(f"Resuming from existing stats file: {stats_file}")
        df = pd.read_csv(stats_file)
    else:
        logger.info(f"Fetching stats for season {season}...")
        # Dummy data
        df = pd.DataFrame({
            "player_id": [1,2,3],
            "player_name": ["Player A", "Player B", "Player C"],
            "team": ["Team X", "Team Y", "Team Z"],
            "points_avg": [25.3, 18.2, 12.5],
            "assists_avg": [5.1, 7.2, 3.8]
        })
        df.to_csv(stats_file, index=False)
        logger.info(f"Saved player stats to {stats_file}")

    return df

if __name__ == "__main__":
    main()
