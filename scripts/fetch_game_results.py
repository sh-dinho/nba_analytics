# ============================================================
# File: scripts/fetch_game_results.py
# Purpose: Fetch NBA game results (today or yesterday) + upcoming schedule
# ============================================================

import argparse
import datetime
import shutil
from pathlib import Path
import requests
import pandas as pd

from core.paths import GAME_RESULTS_FILE, ARCHIVE_DIR, ensure_dirs
from core.log_config import init_global_logger
from core.exceptions import DataError, FileError

logger = init_global_logger()

REQUIRED_COLUMNS = [
    "game_id", "date", "home_team", "away_team",
    "home_score", "away_score", "home_win"
]

# NBA mobile JSON feeds (season component should be updated yearly)
NBA_SEASON_YEAR = "2025"  # adjust when season changes
NBA_SCHEDULE_URL = f"https://data.nba.com/data/v2015/json/mobile_teams/nba/{NBA_SEASON_YEAR}/scores/00_todays_scores.json"

def archive_game_results():
    """Archive existing game results file before overwriting."""
    if GAME_RESULTS_FILE.exists():
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_file = ARCHIVE_DIR / f"game_results_{ts}.csv"
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy(GAME_RESULTS_FILE, archive_file)
        logger.info(f"📦 Archived game results to {archive_file}")

def _iter_games(scoreboard: dict) -> list[dict]:
    """Return a list of game dicts, handling list/dict forms of gs.g."""
    gnode = scoreboard.get("gs", {}).get("g", [])
    if isinstance(gnode, dict):
        return [gnode]
    return gnode or []

def fetch_results_for_date(date: datetime.date, timeout: int = 30) -> pd.DataFrame:
    """Fetch NBA results for a given date from NBA.com feed."""
    try:
        resp = requests.get(NBA_SCHEDULE_URL, timeout=timeout)
        resp.raise_for_status()
        scoreboard = resp.json()

        games = []
        for g in _iter_games(scoreboard):
            game_date = g.get("gdte")
            if game_date != date.strftime("%Y-%m-%d"):
                continue

            # Scores may be empty for in-progress or scheduled games
            home_score_raw = g.get("h", {}).get("s")
            away_score_raw = g.get("v", {}).get("s")

            # Convert to ints if possible
            try:
                home_score = int(home_score_raw) if home_score_raw not in (None, "", " ") else None
            except Exception:
                home_score = None
            try:
                away_score = int(away_score_raw) if away_score_raw not in (None, "", " ") else None
            except Exception:
                away_score = None

            games.append({
                "game_id": g.get("gid"),
                "date": game_date,
                "home_team": g.get("h", {}).get("ta"),
                "away_team": g.get("v", {}).get("ta"),
                "home_score": home_score,
                "away_score": away_score,
                "home_win": None if home_score is None or away_score is None else int(home_score > away_score),
            })

        df = pd.DataFrame(games)
        # Ensure schema
        for col in REQUIRED_COLUMNS:
            if col not in df.columns:
                df[col] = None
        return df[REQUIRED_COLUMNS] if not df.empty else pd.DataFrame(columns=REQUIRED_COLUMNS)
    except requests.HTTPError as e:
        raise DataError(f"HTTP error while fetching results for {date}: {e}")
    except requests.RequestException as e:
        raise DataError(f"Network error while fetching results for {date}: {e}")
    except Exception as e:
        raise DataError(f"Failed to fetch results for {date}: {e}")

def fetch_upcoming_games(timeout: int = 30) -> pd.DataFrame:
    """Fetch upcoming scheduled NBA games (scores not yet available)."""
    try:
        resp = requests.get(NBA_SCHEDULE_URL, timeout=timeout)
        resp.raise_for_status()
        scoreboard = resp.json()

        games = []
        for g in _iter_games(scoreboard):
            status = g.get("stt")
            if status in ("Scheduled", "Pre-Game"):
                games.append({
                    "game_id": g.get("gid"),
                    "date": g.get("gdte"),
                    "home_team": g.get("h", {}).get("ta"),
                    "away_team": g.get("v", {}).get("ta"),
                    "home_score": None,
                    "away_score": None,
                    "home_win": None,
                })

        df = pd.DataFrame(games)
        for col in REQUIRED_COLUMNS:
            if col not in df.columns:
                df[col] = None
        return df[REQUIRED_COLUMNS] if not df.empty else pd.DataFrame(columns=REQUIRED_COLUMNS)
    except requests.HTTPError as e:
        logger.warning(f"⚠️ HTTP error while fetching upcoming games: {e}")
        return pd.DataFrame(columns=REQUIRED_COLUMNS)
    except requests.RequestException as e:
        logger.warning(f"⚠️ Network error while fetching upcoming games: {e}")
        return pd.DataFrame(columns=REQUIRED_COLUMNS)
    except Exception as e:
        logger.warning(f"⚠️ Failed to fetch upcoming games: {e}")
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

def validate_results(df: pd.DataFrame):
    """Ensure the DataFrame has the required schema."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise DataError(f"Missing required columns in game results: {missing}")

def main(timeout: int = 30) -> pd.DataFrame:
    try:
        ensure_dirs(strict=False)
        today = datetime.date.today()
        yesterday = today - datetime.timedelta(days=1)

        logger.info(f"📡 Fetching NBA results for {today}...")
        df_today = fetch_results_for_date(today, timeout=timeout)

        df = df_today
        if df_today.empty:
            logger.info(f"No finalized results for {today}, falling back to {yesterday}...")
            df_yest = fetch_results_for_date(yesterday, timeout=timeout)
            df = df_yest

        # Upcoming schedule
        upcoming_df = fetch_upcoming_games(timeout=timeout)
        if not upcoming_df.empty:
            df = pd.concat([df, upcoming_df], ignore_index=True)

        validate_results(df)

        # Persist with archiving
        archive_game_results()
        GAME_RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(GAME_RESULTS_FILE, index=False)

        logger.info(f"✅ Game results saved to {GAME_RESULTS_FILE} with {len(df)} rows")
        return df

    except DataError as e:
        logger.error(f"❌ {e}")
        raise
    except Exception as e:
        msg = f"Failed to fetch game results: {e}"
        logger.error(f"❌ {msg}")
        raise DataError(msg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch NBA game results")
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout in seconds")
    args = parser.parse_args()
    main(timeout=args.timeout)
