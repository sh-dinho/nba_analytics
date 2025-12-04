# ============================================================
# File: scripts/fetch_player_stats.py
# Purpose: Fetch player stats for the current NBA season (real data)
# ============================================================

import argparse
import pandas as pd
import datetime
import shutil
from nba_api.stats.endpoints import leaguedashplayerstats

from core.paths import DATA_DIR, ARCHIVE_DIR, PLAYER_STATS_FILE, ensure_dirs
from core.log_config import init_global_logger
from core.exceptions import FileError

logger = init_global_logger()

def archive_player_stats():
    """Archive existing player stats file before overwriting."""
    if PLAYER_STATS_FILE.exists():
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_file = ARCHIVE_DIR / f"player_stats_{ts}.csv"
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy(PLAYER_STATS_FILE, archive_file)
        logger.info(f"📦 Archived player stats to {archive_file}")

def fetch_live_player_stats(season: str) -> pd.DataFrame:
    """
    Fetch real player stats from NBA.com using nba_api.
    Season format must be 'YYYY-YY', e.g. '2025-26'.
    """
    logger.info(f"Fetching player stats for season {season} from NBA API...")
    try:
        stats = leaguedashplayerstats.LeagueDashPlayerStats(season=season)
        df = stats.get_data_frames()[0]
        logger.info(f"✅ Retrieved {len(df)} player rows from NBA API for season {season}")
        return df
    except Exception as e:
        raise FileError(f"Failed to fetch player stats for season {season}", file_path=str(PLAYER_STATS_FILE)) from e

def main(season: str, force_refresh: bool = False):
    ensure_dirs(strict=False)

    if PLAYER_STATS_FILE.exists() and not force_refresh:
        logger.info(f"Player stats already exist at {PLAYER_STATS_FILE}. Skipping fetch.")
        return

    try:
        df = fetch_live_player_stats(season)
        archive_player_stats()
        PLAYER_STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(PLAYER_STATS_FILE, index=False)
        logger.info(f"💾 Player stats saved to {PLAYER_STATS_FILE} ({len(df)} rows)")
    except FileError as e:
        logger.error(f"❌ {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch player stats for NBA season")
    parser.add_argument("--season", type=str, required=True,
                        help="NBA season string, e.g. '2025-26'")
    parser.add_argument("--force_refresh", action="store_true",
                        help="Force re-fetch even if file exists")
    args = parser.parse_args()

    main(season=args.season, force_refresh=args.force_refresh)
