# ============================================================
# File: scripts/ingest_games.py
# Purpose: Unified ingestion of NBA games, team stats, and odds with retries, compression, archiving, and summary logging
# ============================================================

import pandas as pd
import datetime
import time
from pathlib import Path
import argparse
import os
import requests

from nba_api.stats.endpoints import leaguegamefinder, teamdashboardbygeneralsplits
from nba_core.paths import DATA_DIR, HISTORICAL_GAMES_FILE, DOWNLOAD_SUMMARY_FILE, ARCHIVE_DIR, ensure_dirs
from nba_core.log_config import init_global_logger
from nba_core.exceptions import FileError, DataError

logger = init_global_logger("ingest_games")

def _season_filename(season: str, gzip: bool) -> Path:
    suffix = ".csv.gz" if gzip else ".csv"
    return DATA_DIR / f"games_{season}{suffix}"

def _archive_file(path: Path, prefix: str):
    if path.exists():
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        archive_path = ARCHIVE_DIR / f"{prefix}_{ts}{''.join(path.suffixes)}"
        archive_path.write_bytes(path.read_bytes())
        logger.info(f"📦 Archived {path.name} → {archive_path}")

def fetch_season(season: str, retries: int = 3, delay: int = 5) -> pd.DataFrame | None:
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

def _parse_seasons(season_start: str, season_end: str | None, only: str | None, update: bool) -> list[str]:
    today = datetime.date.today()
    year = today.year
    latest_season = f"{year-1}-{str(year)[-2:]}" if today.month < 10 else f"{year}-{str(year+1)[-2:]}"
    if update:
        return [latest_season]
    if only:
        return [only]
    end_year = int(season_end.split("-")[0]) if season_end else int(latest_season.split("-")[0])
    start_year = int(season_start.split("-")[0])
    return [f"{y}-{str(y + 1)[-2:]}" for y in range(start_year, end_year + 1)]

def _fetch_team_stats_current(season_label: str) -> pd.DataFrame:
    endpoint = teamdashboardbygeneralsplits.TeamDashboardByGeneralSplits(season=season_label)
    df = endpoint.get_data_frames()[0]
    df["season"] = season_label
    logger.info(f"✅ Fetched team stats for {season_label} ({len(df)} rows)")
    return df

def _fetch_odds_snapshot() -> pd.DataFrame:
    api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        logger.warning("⚠️ ODDS_API_KEY not set; skipping odds snapshot")
        return pd.DataFrame()
    url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/odds/?regions=us&markets=h2h,totals&apiKey={api_key}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        records = []
        for game in data:
            home = game.get("home_team")
            away = game.get("away_team")
            for bookmaker in game.get("bookmakers", []):
                for market in bookmaker.get("markets", []):
                    if market.get("key") == "totals":
                        for outcome in market.get("outcomes", []):
                            records.append({
                                "game": f"{home} vs {away}",
                                "ou_line": outcome.get("point"),
                                "american_odds": outcome.get("price"),
                                "bookmaker": bookmaker.get("title")
                            })
        df = pd.DataFrame(records)
        logger.info(f"✅ Fetched {len(df)} odds records")
        return df
    except Exception as e:
        logger.warning(f"⚠️ Odds snapshot fetch failed: {e}")
        return pd.DataFrame()

def main(season_start: str, season_end: str | None = None, only: str | None = None,
         force: bool = False, gzip: bool = False, update: bool = False,
         retries: int = 3, delay: int = 5, cli_args: dict | None = None):
    ensure_dirs(strict=False)
    seasons = _parse_seasons(season_start, season_end, only, update)
    logger.info(f"🗓️ Target seasons: {', '.join(seasons)}")

    all_games: list[pd.DataFrame] = []
    summary_entries: list[dict] = []

    for season in seasons:
        season_file = _season_filename(season, gzip=gzip)
        if season_file.exists() and not force:
            try:
                df = pd.read_csv(season_file)
                logger.info(f"⏩ Skipping {season}, already downloaded → {season_file}")
                all_games.append(df)
                summary_entries.append({"season": season, "status": "skipped", "rows": len(df)})
            except Exception as e:
                raise FileError(f"Failed to read existing season file {season_file}", file_path=str(season_file)) from e
            continue

        df = fetch_season(season, retries=retries, delay=delay)
        if df is not None:
            all_games.append(df)
            _archive_file(season_file, prefix=f"games_{season}")
            try:
                season_file.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(season_file, index=False, compression="gzip" if gzip else None)
                status = "downloaded" if not force else "re-downloaded"
                size_kb = season_file.stat().st_size / 1024
                logger.info(f"📂 Saved season file → {season_file} ({len(df)} rows, {size_kb:.1f} KB)")
                summary_entries.append({"season": season, "status": status, "rows": len(df)})
            except Exception as e:
                raise FileError(f"Failed to write season file {season_file}", file_path=str(season_file)) from e
        else:
            summary_entries.append({"season": season, "status": "failed", "rows": 0})

    if not all_games:
        raise DataError("No games downloaded for the requested seasons.")

    try:
        if HISTORICAL_GAMES_FILE.exists():
            existing = pd.read_csv(HISTORICAL_GAMES_FILE)
            combined = pd.concat([existing, *all_games], ignore_index=True).drop_duplicates()
        else:
            combined = pd.concat(all_games, ignore_index=True)
    except Exception as e:
        raise FileError(f"Failed to read historical dataset {HISTORICAL_GAMES_FILE}", file_path=str(HISTORICAL_GAMES_FILE)) from e

    _archive_file(HISTORICAL_GAMES_FILE, prefix="historical_games")
    try:
        HISTORICAL_GAMES_FILE.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(HISTORICAL_GAMES_FILE, index=False, compression="gzip" if gzip else None)
        logger.info(f"✅ Combined dataset saved → {HISTORICAL_GAMES_FILE} ({len(combined)} rows)")
    except Exception as e:
        raise FileError(f"Failed to write historical dataset {HISTORICAL_GAMES_FILE}", file_path=str(HISTORICAL_GAMES_FILE)) from e

    run_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary_df = pd.DataFrame(summary_entries)
    summary_df["timestamp"] = run_time
    summary_df["total_rows"] = len(combined)
    if cli_args:
        summary_df["args"] = str(cli_args)
    try:
        DOWNLOAD_SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(DOWNLOAD_SUMMARY_FILE, mode="a", header=not DOWNLOAD_SUMMARY_FILE.exists(), index=False)
        logger.info(f"📈 Download summary appended to {DOWNLOAD_SUMMARY_FILE}")
    except Exception as e:
        logger.warning(f"⚠️ Failed to append download summary: {e}")

    today = datetime.date.today().strftime("%Y-%m-%d")
    latest = seasons[-1]
    stats_df = _fetch_team_stats_current(latest)
    if not stats_df.empty:
        stats_path = DATA_DIR / f"nba_team_stats_{latest.replace('-', '')}_{today}.csv"
        stats_df.to_csv(stats_path, index=False)
        logger.info(f"📦 Saved team stats → {stats_path}")
    odds_df = _fetch_odds_snapshot()
    if not odds_df.empty:
        odds_path = DATA_DIR / f"nba_odds_{latest.replace('-', '')}_{today}.csv"
        odds_df.to_csv(odds_path, index=False)
        logger.info(f"🎲 Saved odds snapshot → {odds_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified NBA ingestion")
    parser.add_argument("--season_start", type=str, default="2020-21")
    parser.add_argument("--season_end", type=str, default=None)
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--gzip", action="store_true")
    parser.add_argument("--update", action="store_true")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--delay", type=int, default=5)
    args = parser.parse_args()
    main(**vars(args))
