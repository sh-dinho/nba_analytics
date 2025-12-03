# ============================================================
# File: scripts/fetch_game_results.py
# Purpose: Fetch NBA game results (today or yesterday) + upcoming schedule
# ============================================================

import argparse
import pandas as pd
import datetime
import requests
from pathlib import Path
from core.config import GAME_RESULTS_FILE, ensure_dirs
from core.log_config import setup_logger
from core.exceptions import DataError

logger = setup_logger("fetch_game_results")

REQUIRED_COLUMNS = [
    "game_id", "date", "home_team", "away_team",
    "home_score", "away_score", "home_win"
]

# NBA.com JSON feed (example endpoint for scoreboard)
NBA_SCOREBOARD_URL = "https://data.nba.com/data/v2015/json/mobile_teams/nba/2025/scores/gamedetail/{game_id}_gamedetail.json"
NBA_SCHEDULE_URL = "https://data.nba.com/data/v2015/json/mobile_teams/nba/2025/scores/00_todays_scores.json"


def fetch_results_for_date(date: datetime.date) -> pd.DataFrame:
    """Fetch NBA results for a given date from NBA.com feed."""
    try:
        # NBA.com provides a daily scoreboard JSON
        url = NBA_SCHEDULE_URL
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()

        games = []
        for g in data.get("gs", {}).get("g", []):
            game_date = g.get("gdte")
            if game_date != date.strftime("%Y-%m-%d"):
                continue

            home_score = int(g.get("h", {}).get("s", 0))
            away_score = int(g.get("v", {}).get("s", 0))
            games.append({
                "game_id": g.get("gid"),
                "date": game_date,
                "home_team": g.get("h", {}).get("ta"),
                "away_team": g.get("v", {}).get("ta"),
                "home_score": home_score if home_score else None,
                "away_score": away_score if away_score else None,
                "home_win": None if not home_score or not away_score else int(home_score > away_score)
            })

        return pd.DataFrame(games, columns=REQUIRED_COLUMNS)
    except Exception as e:
        raise DataError(f"Failed to fetch results for {date}: {e}")


def fetch_upcoming_games() -> pd.DataFrame:
    """Fetch upcoming scheduled NBA games (scores not yet available)."""
    try:
        response = requests.get(NBA_SCHEDULE_URL, timeout=30)
        response.raise_for_status()
        data = response.json()

        games = []
        for g in data.get("gs", {}).get("g", []):
            if g.get("stt") in ["Scheduled", "Pre-Game"]:
                games.append({
                    "game_id": g.get("gid"),
                    "date": g.get("gdte"),
                    "home_team": g.get("h", {}).get("ta"),
                    "away_team": g.get("v", {}).get("ta"),
                    "home_score": None,
                    "away_score": None,
                    "home_win": None
                })

        return pd.DataFrame(games, columns=REQUIRED_COLUMNS)
    except Exception as e:
        logger.warning(f"⚠️ Failed to fetch upcoming games: {e}")
        return pd.DataFrame(columns=REQUIRED_COLUMNS)


def validate_results(df: pd.DataFrame):
    """Ensure the DataFrame has the required schema."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise DataError(f"Missing required columns in game results: {missing}")


def main():
    try:
        ensure_dirs()
        today = datetime.date.today()
        yesterday = today - datetime.timedelta(days=1)

        logger.info(f"Fetching NBA results for {today}...")
        df = fetch_results_for_date(today)

        if df.empty:
            logger.info(f"No results for {today}, falling back to {yesterday}...")
            df = fetch_results_for_date(yesterday)

        upcoming_df = fetch_upcoming_games()
        if not upcoming_df.empty:
            df = pd.concat([df, upcoming_df], ignore_index=True)

        validate_results(df)

        Path(GAME_RESULTS_FILE).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(GAME_RESULTS_FILE, index=False)

        logger.info(f"✅ Game results saved to {GAME_RESULTS_FILE} with {len(df)} rows")
        return df

    except Exception as e:
        logger.error(f"❌ Failed to fetch game results: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch NBA game results")
    args = parser.parse_args()
    main()