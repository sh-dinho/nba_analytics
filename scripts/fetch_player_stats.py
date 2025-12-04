# ============================================================
# File: scripts/fetch_player_stats.py
# Purpose: Fetch player stats for the current NBA season (real data)
# Features: retries, archiving, retention, dry-run, append mode
# ============================================================

import argparse
import pandas as pd
import datetime
import shutil
import os
import time
from pathlib import Path
from nba_api.stats.endpoints import leaguedashplayerstats

from core.paths import DATA_DIR, ARCHIVE_DIR, PLAYER_STATS_FILE, ensure_dirs
from core.log_config import init_global_logger
from core.exceptions import FileError

logger = init_global_logger()

ARCHIVE_RETENTION = 5  # Keep last 5 archives


def get_current_season() -> str:
    """Return current NBA season string in 'YYYY-YY' format."""
    today = datetime.date.today()
    start_year = today.year - 1 if today.month < 10 else today.year
    end_year = str(start_year + 1)[-2:]
    return f"{start_year}-{end_year}"


def archive_player_stats():
    """Archive existing player stats file before overwriting and enforce retention."""
    if PLAYER_STATS_FILE.exists():
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_file = ARCHIVE_DIR / f"player_stats_{ts}.csv"
        shutil.copy(PLAYER_STATS_FILE, archive_file)
        logger.info(f"📦 Archived player stats to {archive_file}")

        # Cleanup old archives beyond retention
        archives = sorted(ARCHIVE_DIR.glob("player_stats_*.csv"), reverse=True)
        for old_file in archives[ARCHIVE_RETENTION:]:
            old_file.unlink()
            logger.info(f"🗑️ Removed old archive {old_file}")


def fetch_live_player_stats(season: str, retries: int = 3, delay: int = 5) -> pd.DataFrame:
    """Fetch real player stats from NBA.com using nba_api, with retries."""
    logger.info(f"Fetching player stats for season {season} from NBA API...")
    attempt = 0
    while attempt < retries:
        try:
            stats = leaguedashplayerstats.LeagueDashPlayerStats(season=season)
            df = stats.get_data_frames()[0]
            if df.empty:
                logger.warning(f"⚠️ No data returned for season {season}")
            logger.info(f"✅ Retrieved {len(df)} player rows for season {season}")
            return df
        except Exception as e:
            attempt += 1
            logger.warning(f"Attempt {attempt}/{retries} failed: {e}")
            if attempt < retries:
                time.sleep(delay)
            else:
                raise FileError(f"Failed to fetch player stats for season {season}") from e


def main(
    season: str,
    force_refresh: bool = False,
    columns: list[str] = None,
    output: str = None,
    file_format: str = "csv",
    dry_run: bool = False,
    append: bool = False
):
    ensure_dirs(strict=False)
    df = fetch_live_player_stats(season)

    # Filter columns if provided
    if columns:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            logger.warning(f"⚠️ Some requested columns not found: {missing}")
        df = df[[c for c in columns if c in df.columns]]

    if df.empty:
        logger.warning("⚠️ Fetched DataFrame is empty. Exiting.")
        return

    # Dry-run mode
    if dry_run:
        logger.info(f"📝 Dry-run: DataFrame shape: {df.shape}, columns: {list(df.columns)}")
        return

    # Archive existing file if it exists and force_refresh is True
    if force_refresh or not PLAYER_STATS_FILE.exists():
        archive_player_stats()

    # Decide output path
    save_path = DATA_DIR / output if output else PLAYER_STATS_FILE

    # Handle append mode
    if append and save_path.exists():
        existing = pd.read_csv(save_path)
        df = pd.concat([existing, df], ignore_index=True)
        logger.info(f"🔄 Appending to existing file, new shape: {df.shape}")

    # If file exists and not force_refresh, append timestamp
    if save_path.exists() and not force_refresh and not append:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        stem, ext = os.path.splitext(save_path.name)
        save_path = save_path.parent / f"{stem}_{ts}{ext}"
        logger.info(f"⚠️ File exists. Saving with timestamp: {save_path}")

    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Save in chosen format
    if file_format == "csv":
        df.to_csv(save_path, index=False)
    elif file_format == "json":
        df.to_json(save_path, orient="records", indent=2)
    elif file_format == "parquet":
        df.to_parquet(save_path, index=False)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")

    logger.info(f"💾 Player stats saved to {save_path} ({len(df)} rows, {len(df.columns)} columns)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch player stats for NBA season")
    parser.add_argument("--season", type=str, required=False,
                        help="NBA season string, e.g. '2025-26'. Defaults to current season.")
    parser.add_argument("--force_refresh", action="store_true",
                        help="Force re-fetch even if file exists")
    parser.add_argument("--append", action="store_true",
                        help="Append fetched stats to existing file instead of overwriting")
    parser.add_argument("--columns", nargs="+", type=str,
                        default=["PLAYER_NAME", "TEAM_ABBREVIATION", "PTS", "REB", "AST"],
                        help="Columns to save, e.g. --columns PLAYER_NAME TEAM_ABBREVIATION PTS REB AST")
    parser.add_argument("--output", type=str, required=False,
                        help="Custom output filename, e.g. player_stats_2025.csv")
    parser.add_argument("--format", type=str, choices=["csv", "json", "parquet"], default="csv",
                        help="Output file format: csv, json, parquet")
    parser.add_argument("--dry_run", action="store_true",
                        help="Preview DataFrame without saving")
    args = parser.parse_args()

    season = args.season if args.season else get_current_season()
    main(
        season=season,
        force_refresh=args.force_refresh,
        columns=args.columns,
        output=args.output,
        file_format=args.format,
        dry_run=args.dry_run,
        append=args.append
    )
