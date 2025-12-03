# ============================================================
# File: scripts/fetch_player_stats.py
# Purpose: Fetch player stats for the current NBA season (real data)
# ============================================================

import argparse
import os
import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats
from core.config import PLAYER_STATS_FILE
from core.log_config import setup_logger

logger = setup_logger("fetch_player_stats")


def fetch_live_player_stats(season: str) -> pd.DataFrame:
    """
    Fetch real player stats from NBA.com using nba_api.
    Season format must be 'YYYY-YY', e.g. '2025-26'.
    """
    logger.info(f"Fetching player stats for season {season} from NBA API...")

    # Query NBA API for player stats
    stats = leaguedashplayerstats.LeagueDashPlayerStats(season=season)
    df = stats.get_data_frames()[0]  # returns a list of DataFrames

    logger.info(f"Retrieved {len(df)} player rows from NBA API for season {season}")
    return df


def main(season: str, force_refresh: bool = False):
    if os.path.exists(PLAYER_STATS_FILE) and not force_refresh:
        logger.info(f"Player stats already exist at {PLAYER_STATS_FILE}. Skipping fetch.")
        return

    try:
        df = fetch_live_player_stats(season)
        df.to_csv(PLAYER_STATS_FILE, index=False)
        logger.info(f"✅ Player stats saved to {PLAYER_STATS_FILE} ({len(df)} rows)")
    except Exception as e:
        logger.error(f"❌ Failed to fetch player stats: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch player stats for NBA season")
    parser.add_argument("--season", type=str, required=True,
                        help="NBA season string, e.g. '2025-26'")
    parser.add_argument("--force_refresh", action="store_true",
                        help="Force re-fetch even if file exists")
    args = parser.parse_args()

    main(season=args.season, force_refresh=args.force_refresh)