# ============================================================
# File: scripts/fetch_player_stats.py
# Purpose: Fetch player stats for a given NBA season
# ============================================================

import argparse
import pandas as pd
from core.config import PLAYER_STATS_FILE, BASE_DATA_DIR
from core.log_config import setup_logger

logger = setup_logger("fetch_player_stats")


def fetch_player_stats(season: str = "2024-25", use_synthetic: bool = False):
    """
    Fetch player stats for the given season.
    If use_synthetic=True, generate synthetic data instead of fetching live.
    Saves results to PLAYER_STATS_FILE.
    """
    logger.info(f"Fetching player stats for season {season} (synthetic={use_synthetic})...")

    if use_synthetic:
        # Example synthetic dataset
        data = {
            "player": ["LeBron James", "Stephen Curry", "Jayson Tatum"],
            "team": ["LAL", "GSW", "BOS"],
            "season": [season] * 3,
            "points": [27.2, 29.5, 26.8],
            "assists": [7.4, 6.3, 4.4],
            "rebounds": [8.1, 5.2, 8.0],
        }
        df = pd.DataFrame(data)
    else:
        # 🚀 Stub for live fetch logic
        # Replace this with actual API calls (e.g., NBA stats API, sportsdata.io, balldontlie.io)
        try:
            # Example placeholder: simulate API response
            logger.info("Attempting live fetch (stub)...")
            # TODO: Replace with requests.get(...) to NBA API
            data = {
                "player": ["Giannis Antetokounmpo", "Luka Doncic", "Nikola Jokic"],
                "team": ["MIL", "DAL", "DEN"],
                "season": [season] * 3,
                "points": [28.1, 30.2, 26.4],
                "assists": [5.9, 8.1, 7.0],
                "rebounds": [11.0, 9.2, 12.3],
            }
            df = pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Live fetch failed: {e}. Falling back to synthetic data.")
            data = {
                "player": ["LeBron James", "Stephen Curry", "Jayson Tatum"],
                "team": ["LAL", "GSW", "BOS"],
                "season": [season] * 3,
                "points": [27.2, 29.5, 26.8],
                "assists": [7.4, 6.3, 4.4],
                "rebounds": [8.1, 5.2, 8.0],
            }
            df = pd.DataFrame(data)

    # Ensure directory exists
    BASE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    df.to_csv(PLAYER_STATS_FILE, index=False)
    logger.info(f"✅ Player stats saved to {PLAYER_STATS_FILE} ({len(df)} rows)")


def main():
    parser = argparse.ArgumentParser(description="Fetch player stats")
    parser.add_argument("--season", type=str, default="2024-25",
                        help="Season to fetch stats for (e.g. 2024-25)")
    parser.add_argument("--use_synthetic", action="store_true",
                        help="Use synthetic data instead of live fetch")
    args = parser.parse_args()

    fetch_player_stats(season=args.season, use_synthetic=args.use_synthetic)


if __name__ == "__main__":
    main()