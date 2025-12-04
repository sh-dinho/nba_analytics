# ============================================================
# File: scripts/download_games.py
# Purpose: Download NBA games from a chosen start season to current
#          Save both per-season files and one combined dataset
#          Skip already-downloaded seasons unless --force is used
#          Support --only, --update, retries, compression, and summary logging
# ============================================================

import pandas as pd
import datetime
import time
from pathlib import Path
import argparse

from nba_api.stats.endpoints import leaguegamefinder

from core.paths import DATA_DIR, HISTORICAL_GAMES_FILE, DOWNLOAD_SUMMARY_FILE, ARCHIVE_DIR, ensure_dirs
from core.log_config import init_global_logger
from core.exceptions import FileError, DataError

logger = init_global_logger()

def _season_filename(season: str, gzip: bool) -> Path:
    suffix = ".csv.gz" if gzip else ".csv"
    return DATA_DIR / f"games_{season}{suffix}"

def _archive_file(path: Path, prefix: str):
    """Archive an existing file to data/archive with timestamp."""
    if path.exists():
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        archive_path = ARCHIVE_DIR / f"{prefix}_{ts}{''.join(path.suffixes)}"
        archive_path.write_bytes(path.read_bytes())
        logger.info(f"📦 Archived {path.name} → {archive_path}")

def fetch_season(season: str, retries: int = 3, delay: int = 5) -> pd.DataFrame | None:
    """Fetch one season with retries."""
    for attempt in range(1, retries + 1):
        try:
            gf = leaguegamefinder.LeagueGameFinder(season_nullable=season)
            df = gf.get_data_frames()[0]
            df["season"] = season
            logger.info(f"✅ Fetched {len(df)} games for {season}")
            return df
        except Exception as e:
            logger.warning(f"⚠️ Attempt {attempt}/{retries} failed for {season}: {e}")
            if attempt < retries:
                time.sleep(delay)
    logger.error(f"❌ All retries failed for {season}")
    return None

def download_games(seasons: list[str], force: bool = False, gzip: bool = False) -> tuple[pd.DataFrame, list[dict]]:
    """Download NBA games for given seasons list."""
    all_games: list[pd.DataFrame] = []
    summary_entries: list[dict] = []

    total = len(seasons)
    for idx, season in enumerate(seasons, start=1):
        season_file = _season_filename(season, gzip=gzip)
        logger.info(f"➡️ Processing {season} ({idx}/{total})")

        if season_file.exists() and not force:
            logger.info(f"⏩ Skipping {season}, already downloaded → {season_file}")
            try:
                df = pd.read_csv(season_file)
            except Exception as e:
                raise FileError(f"Failed to read existing season file {season_file}", file_path=str(season_file)) from e
            all_games.append(df)
            summary_entries.append({"season": season, "status": "skipped", "rows": len(df)})
            continue

        df = fetch_season(season)
        if df is not None:
            all_games.append(df)
            # Archive existing per-season file before overwriting
            _archive_file(season_file, prefix=f"games_{season}")
            try:
                season_file.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(season_file, index=False, compression="gzip" if gzip else None)
                status = "downloaded" if not force else "re-downloaded"
                logger.info(f"📂 Saved season file → {season_file}")
                summary_entries.append({"season": season, "status": status, "rows": len(df)})
            except Exception as e:
                raise FileError(f"Failed to write season file {season_file}", file_path=str(season_file)) from e
        else:
            summary_entries.append({"season": season, "status": "failed", "rows": 0})

    if not all_games:
        raise DataError("No games downloaded for the requested seasons.")

    return pd.concat(all_games, ignore_index=True), summary_entries

def _parse_seasons(season_start: str, season_end: str | None, only: str | None, update: bool) -> list[str]:
    """Build the season list based on CLI flags."""
    current_year = datetime.datetime.now().year
    latest_season = f"{current_year-1}-{str(current_year)[-2:]}"

    if update:
        return [latest_season]
    if only:
        return [only]

    # Inclusive start, exclusive end to match year-year+1 format
    end_year = int(season_end.split("-")[0]) if season_end else current_year
    start_year = int(season_start.split("-")[0])
    return [f"{year}-{str(year + 1)[-2:]}" for year in range(start_year, end_year)]

def main(season_start: str, season_end: str | None = None, only: str | None = None,
         force: bool = False, gzip: bool = False, update: bool = False,
         retries: int = 3, delay: int = 5):
    ensure_dirs(strict=False)

    seasons = _parse_seasons(season_start, season_end, only, update)
    logger.info(f"🗓️ Target seasons: {', '.join(seasons)}")

    new_data, summary_entries = download_games(seasons, force=force, gzip=gzip)

    # Load existing dataset if present, then merge and de-duplicate
    try:
        if HISTORICAL_GAMES_FILE.exists():
            existing = pd.read_csv(HISTORICAL_GAMES_FILE)
            combined = pd.concat([existing, new_data], ignore_index=True).drop_duplicates()
        else:
            combined = new_data
    except Exception as e:
        raise FileError(f"Failed to read historical dataset {HISTORICAL_GAMES_FILE}", file_path=str(HISTORICAL_GAMES_FILE)) from e

    # Archive before saving combined dataset
    _archive_file(HISTORICAL_GAMES_FILE, prefix="historical_games")

    try:
        HISTORICAL_GAMES_FILE.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(HISTORICAL_GAMES_FILE, index=False, compression="gzip" if gzip else None)
        logger.info(f"✅ Combined dataset saved → {HISTORICAL_GAMES_FILE} ({len(combined)} rows)")
    except Exception as e:
        raise FileError(f"Failed to write historical dataset {HISTORICAL_GAMES_FILE}", file_path=str(HISTORICAL_GAMES_FILE)) from e

    # Append summary log (CSV in logs directory)
    run_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary_df = pd.DataFrame(summary_entries)
    summary_df["timestamp"] = run_time
    summary_df["total_rows"] = len(combined)
    try:
        DOWNLOAD_SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
        if DOWNLOAD_SUMMARY_FILE.exists():
            summary_df.to_csv(DOWNLOAD_SUMMARY_FILE, mode="a", header=False, index=False)
        else:
            summary_df.to_csv(DOWNLOAD_SUMMARY_FILE, index=False)
        logger.info(f"📈 Download summary appended to {DOWNLOAD_SUMMARY_FILE}")
    except Exception as e:
        logger.warning(f"⚠️ Failed to append download summary: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download NBA games and append to dataset")
    parser.add_argument("--season_start", type=str, default="2020-21",
                        help="Starting season (format: YYYY-YY, e.g., 2015-16)")
    parser.add_argument("--season_end", type=str, default=None,
                        help="Ending season (format: YYYY-YY, e.g., 2022-23). Defaults to current season.")
    parser.add_argument("--only", type=str, default=None,
                        help="Download only one specific season (format: YYYY-YY, e.g., 2021-22)")
    parser.add_argument("--force", action="store_true",
                        help="Force re-download even if season file already exists")
    parser.add_argument("--gzip", action="store_true",
                        help="Save files compressed as .csv.gz")
    parser.add_argument("--update", action="store_true",
                        help="Download the latest season only, if missing")
    parser.add_argument("--retries", type=int, default=3,
                        help="Number of retries per season fetch")
    parser.add_argument("--delay", type=int, default=5,
                        help="Delay between retries in seconds")
    args = parser.parse_args()

    main(args.season_start, args.season_end, args.only, args.force, args.gzip, args.update, args.retries, args.delay)
