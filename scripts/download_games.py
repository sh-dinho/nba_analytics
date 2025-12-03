# ============================================================
# File: scripts/download_games.py
# Purpose: Download NBA games from a chosen start season to current
#          Save both per-season files and one combined dataset
#          Skip already-downloaded seasons unless --force is used
#          Support --only, --update, retries, compression, and summary logging
# ============================================================

import pandas as pd
import datetime
import os
import argparse
import time
from nba_api.stats.endpoints import leaguegamefinder
from core.config import HISTORICAL_GAMES_FILE, BASE_DATA_DIR
from core.log_config import setup_logger

logger = setup_logger("download_games")

SUMMARY_LOG = os.path.join(BASE_DATA_DIR, "download_summary.log")

def fetch_season(season: str, retries: int = 3, delay: int = 5):
    """Fetch one season with retries."""
    for attempt in range(1, retries+1):
        try:
            gf = leaguegamefinder.LeagueGameFinder(season_nullable=season)
            df = gf.get_data_frames()[0]
            df["season"] = season
            return df
        except Exception as e:
            logger.warning(f"⚠️ Attempt {attempt}/{retries} failed for {season}: {e}")
            time.sleep(delay)
    logger.error(f"❌ All retries failed for {season}")
    return None

def download_games(seasons, force=False, gzip=False):
    """Download NBA games for given seasons list."""
    all_games = []
    summary_entries = []

    total = len(seasons)
    for idx, season in enumerate(seasons, start=1):
        season_file = os.path.join(BASE_DATA_DIR, f"games_{season}.csv.gz" if gzip else f"games_{season}.csv")
        logger.info(f"➡️ Processing {season} ({idx}/{total})")

        if os.path.exists(season_file) and not force:
            logger.info(f"⏩ Skipping {season}, already downloaded → {season_file}")
            df = pd.read_csv(season_file)
            all_games.append(df)
            summary_entries.append({"season": season, "status": "skipped", "rows": len(df)})
            continue

        df = fetch_season(season)
        if df is not None:
            all_games.append(df)
            # Save per-season file (overwrite if force=True)
            df.to_csv(season_file, index=False, compression="gzip" if gzip else None)
            status = "downloaded" if not force else "re-downloaded"
            logger.info(f"📂 Saved season file → {season_file}")
            summary_entries.append({"season": season, "status": status, "rows": len(df)})
        else:
            summary_entries.append({"season": season, "status": "failed", "rows": 0})

    if not all_games:
        raise ValueError("No games downloaded.")

    return pd.concat(all_games, ignore_index=True), summary_entries

def main(season_start, season_end=None, only=None, force=False, gzip=False, update=False):
    current_year = datetime.datetime.now().year
    latest_season = f"{current_year-1}-{str(current_year)[-2:]}"
    
    if update:
        logger.info(f"🔄 Update mode: checking latest season {latest_season}")
        season_file = os.path.join(BASE_DATA_DIR, f"games_{latest_season}.csv.gz" if gzip else f"games_{latest_season}.csv")
        if os.path.exists(season_file) and not force:
            logger.info(f"⏩ Latest season {latest_season} already downloaded → {season_file}")
            return
        else:
            seasons = [latest_season]
    elif only:
        seasons = [only]
    else:
        end_year = int(season_end.split("-")[0]) if season_end else current_year
        start_year = int(season_start.split("-")[0])
        seasons = [f"{year}-{str(year+1)[-2:]}" for year in range(start_year, end_year)]

    new_data, summary_entries = download_games(seasons, force=force, gzip=gzip)

    # Load existing dataset if present
    if os.path.exists(HISTORICAL_GAMES_FILE):
        existing = pd.read_csv(HISTORICAL_GAMES_FILE)
        combined = pd.concat([existing, new_data], ignore_index=True).drop_duplicates()
    else:
        combined = new_data

    # Save combined dataset
    combined.to_csv(HISTORICAL_GAMES_FILE, index=False, compression="gzip" if gzip else None)
    logger.info(f"✅ Combined dataset saved → {HISTORICAL_GAMES_FILE} ({len(combined)} rows)")

    # Append summary log
    run_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary_df = pd.DataFrame(summary_entries)
    summary_df["timestamp"] = run_time
    summary_df["total_rows"] = len(combined)
    try:
        if os.path.exists(SUMMARY_LOG):
            summary_df.to_csv(SUMMARY_LOG, mode="a", header=False, index=False)
        else:
            summary_df.to_csv(SUMMARY_LOG, index=False)
        logger.info(f"📈 Download summary appended to {SUMMARY_LOG}")
    except Exception as e:
        logger.warning(f"Failed to append download summary: {e}")

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
                        help="Update dataset by downloading the latest season if missing")
    args = parser.parse_args()

    main(args.season_start, args.season_end, args.only, args.force, args.gzip, args.update)