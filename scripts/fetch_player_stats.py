# ============================================================
# File: scripts/fetch_player_stats.py
# Purpose: Fetch player stats for the current NBA season (real data)
# ============================================================

import argparse
import pandas as pd
import datetime
import shutil
import os
from nba_api.stats.endpoints import leaguedashplayerstats

from core.paths import DATA_DIR, ARCHIVE_DIR, PLAYER_STATS_FILE, ensure_dirs
from core.log_config import init_global_logger
from core.exceptions import FileError

logger = init_global_logger()

def get_current_season() -> str:
    """Return the current NBA season string in 'YYYY-YY' format."""
    today = datetime.date.today()
    year = today.year
    # NBA season starts in October, so if before October, use previous year
    if today.month < 10:
        start_year = year - 1
    else:
        start_year = year
    end_year = str(start_year + 1)[-2:]
    return f"{start_year}-{end_year}"

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

def main(season: str, force_refresh: bool = False, columns: list[str] = None, output: str = None, file_format: str = "csv"):
    ensure_dirs(strict=False)

    try:
        df = fetch_live_player_stats(season)

        # Filter columns if provided
        if columns:
            missing = [c for c in columns if c not in df.columns]
            if missing:
                logger.warning(f"⚠️ Some requested columns not found: {missing}")
            df = df[[c for c in columns if c in df.columns]]

        archive_player_stats()

        # Decide output path
        if output:
            save_path = DATA_DIR / output
        else:
            # Default to PLAYER_STATS_FILE
            save_path = PLAYER_STATS_FILE

        # If file exists, append timestamp
        if save_path.exists():
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            stem, ext = os.path.splitext(save_path.name)
            save_path = save_path.parent / f"{stem}_{ts}{ext}"
            logger.info(f"⚠️ File already exists. Saving with timestamp: {save_path}")

        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save in chosen format
        if file_format == "csv":
            df.to_csv(save_path, index=False)
        elif file_format == "json":
            df.to_json(save_path, orient="records", indent=2)
        elif file_format == "parquet":
            df.to_parquet(save_path, index=False)

        logger.info(f"💾 Player stats saved to {save_path} ({len(df)} rows, {len(df.columns)} columns)")
    except FileError as e:
        logger.error(f"❌ {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch player stats for NBA season")
    parser.add_argument("--season", type=str, required=False,
                        help="NBA season string, e.g. '2025-26'. Defaults to current season.")
    parser.add_argument("--force_refresh", action="store_true",
                        help="Force re-fetch even if file exists")
    parser.add_argument("--columns", nargs="+", type=str,
                        default=["PLAYER_NAME", "TEAM_ABBREVIATION", "PTS", "REB", "AST"],
                        help="List of columns to save, e.g. --columns PLAYER_NAME TEAM_ABBREVIATION PTS REB AST")
    parser.add_argument("--output", type=str, required=False,
                        help="Custom output filename, e.g. player_stats_2025.csv")
    parser.add_argument("--format", type=str, choices=["csv", "json", "parquet"], default="csv",
                        help="Output file format: csv, json, or parquet (default: csv)")
    args = parser.parse_args()

    season = args.season if args.season else get_current_season()
    main(season=season, force_refresh=args.force_refresh, columns=args.columns, output=args.output, file_format=args.format)