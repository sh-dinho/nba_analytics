# ============================================================
# File: scripts/fetch_new_games.py
# Purpose: Fetch today's NBA games, player stats, and odds.
# Author: <your name or org>
# Last Updated: 2025-02-21
#
# Notes:
# - Fetches scoreboard + boxscores + odds API data
# - Allows CI-safe placeholder mode (for synthetic CI runs)
# - Produces standardized NEW_GAMES_FILE used by feature builder
# ============================================================

import argparse
import datetime
import json
import shutil
from pathlib import Path
import os
import requests
import pandas as pd
from dotenv import load_dotenv

from core.paths import DATA_DIR, ARCHIVE_DIR, NEW_GAMES_FILE, ensure_dirs
from core.log_config import init_global_logger
from core.exceptions import DataError, FileError

logger = init_global_logger()


REQUIRED_COLUMNS = [
    "PLAYER_NAME", "TEAM_ABBREVIATION", "TEAM_HOME", "TEAM_AWAY",
    "PTS", "AST", "REB", "GAMES_PLAYED", "decimal_odds"
]


# ============================================================
# Season Utilities
# ============================================================

def get_season_year() -> str:
    """Determine NBA season year (season starts in October)."""
    today = datetime.date.today()
    return str(today.year - 1 if today.month < 10 else today.year)


NBA_YEAR = get_season_year()

NBA_SCHEDULE_URL = (
    f"https://data.nba.com/data/v2015/json/mobile_teams/nba/"
    f"{NBA_YEAR}/scores/00_todays_scores.json"
)
BOX_URL_TEMPLATE = (
    f"https://data.nba.com/data/v2015/json/mobile_teams/nba/"
    f"{NBA_YEAR}/scores/gamedetail/{{gid}}_gamedetail.json"
)


# ============================================================
# Odds API Setup
# ============================================================

load_dotenv()
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"


def normalize_team_name(name: str) -> str:
    return name.strip().lower().replace(" ", "").replace("-", "")


# ============================================================
# Archiving
# ============================================================

def archive_new_games():
    if not NEW_GAMES_FILE.exists():
        return
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    dest = ARCHIVE_DIR / f"new_games_{ts}.csv"
    shutil.copy(NEW_GAMES_FILE, dest)
    logger.info(f"📦 Archived previous new_games → {dest}")


# ============================================================
# Odds Fetching
# ============================================================

def fetch_odds(timeout: int = 30) -> dict:
    """Return a dict keyed by (home, away) → {home_odds, away_odds}."""
    if not ODDS_API_KEY:
        logger.warning("⚠️ No ODDS_API_KEY provided. Odds unavailable.")
        return {}

    try:
        params = {
            "apiKey": ODDS_API_KEY,
            "regions": "us",
            "markets": "h2h",
            "oddsFormat": "decimal",
            "dateFormat": "iso",
        }

        resp = requests.get(ODDS_API_URL, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()

        odds_map = {}
        for game in data:
            home = game.get("home_team")
            away = game.get("away_team")
            if not home or not away:
                continue

            home_prices, away_prices = [], []
            for book in game.get("bookmakers", []):
                for market in book.get("markets", []):
                    for outcome in market.get("outcomes", []):
                        name = outcome.get("name")
                        price = outcome.get("price")
                        if name == home and price is not None:
                            home_prices.append(price)
                        elif name == away and price is not None:
                            away_prices.append(price)

            odds_map[(normalize_team_name(home), normalize_team_name(away))] = {
                "home_odds": sum(home_prices)/len(home_prices) if home_prices else None,
                "away_odds": sum(away_prices)/len(away_prices) if away_prices else None,
            }

        logger.info(f"🎲 Odds fetched for {len(odds_map)} games")
        return odds_map

    except Exception as e:
        logger.warning(f"⚠️ Odds API error: {e}")
        return {}


# ============================================================
# Schedule + Boxscore Fetching
# ============================================================

def fetch_schedule(timeout: int):
    """Return scoreboard data and list of today's games."""
    resp = requests.get(NBA_SCHEDULE_URL, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    games_node = data.get("gs", {}).get("g", [])
    games_node = [games_node] if isinstance(games_node, dict) else games_node

    today = datetime.date.today().strftime("%Y-%m-%d")
    games_today = [g for g in games_node if g.get("gdte") == today]

    return games_today


def fetch_boxscore(gid: str, timeout: int):
    """Fetch raw boxscore JSON for a game."""
    url = BOX_URL_TEMPLATE.format(gid=gid)
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


# ============================================================
# Player Parsing
# ============================================================

def parse_player_rows(box_data, home_team, away_team, home_odds, away_odds):
    """Return list of normalized player rows for a game."""
    players = box_data.get("g", {}).get("pstsg", [])

    # No stats yet → placeholder rows
    if not players:
        return [
            {
                "PLAYER_NAME": None,
                "TEAM_ABBREVIATION": home_team,
                "TEAM_HOME": home_team,
                "TEAM_AWAY": away_team,
                "PTS": None, "AST": None, "REB": None, "GAMES_PLAYED": None,
                "decimal_odds": home_odds,
            },
            {
                "PLAYER_NAME": None,
                "TEAM_ABBREVIATION": away_team,
                "TEAM_HOME": home_team,
                "TEAM_AWAY": away_team,
                "PTS": None, "AST": None, "REB": None, "GAMES_PLAYED": None,
                "decimal_odds": away_odds,
            },
        ]

    rows = []
    for p in players:
        team_abbr = p.get("ta")
        player_name = " ".join(filter(None, [p.get("fn"), p.get("ln")]))
        odds = home_odds if team_abbr == home_team else away_odds

        rows.append({
            "PLAYER_NAME": player_name or None,
            "TEAM_ABBREVIATION": team_abbr,
            "TEAM_HOME": home_team,
            "TEAM_AWAY": away_team,
            "PTS": p.get("pts"),
            "AST": p.get("ast"),
            "REB": p.get("reb"),
            "GAMES_PLAYED": p.get("gp"),
            "decimal_odds": odds,
        })

    return rows


# ============================================================
# Schema Enforcement
# ============================================================

def ensure_minimum_schema(df: pd.DataFrame, schedule_empty: bool, allow_placeholder: bool):
    """Ensure all required columns exist and insert placeholders if needed."""
    if df.empty and allow_placeholder:
        placeholder_tag = "NO_GAMES_TODAY" if schedule_empty else "NO_STATS_YET"
        df = pd.DataFrame([{
            "PLAYER_NAME": placeholder_tag,
            "TEAM_ABBREVIATION": None,
            "TEAM_HOME": None,
            "TEAM_AWAY": None,
            "PTS": None, "AST": None, "REB": None, "GAMES_PLAYED": None,
            "decimal_odds": None,
        }])

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = None

    return df[REQUIRED_COLUMNS]


# ============================================================
# Main Fetch Function
# ============================================================

def fetch_new_games(debug=False, allow_placeholder=False, timeout=30) -> Path:
    ensure_dirs(strict=False)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        today = datetime.date.today()
        logger.info(f"📡 Fetching NBA data for {today}...")

        games_today = fetch_schedule(timeout=timeout)
        schedule_empty = len(games_today) == 0

        odds_map = fetch_odds(timeout)
        all_rows = []
        debug_logged = False

        for g in games_today:
            gid = g.get("gid")
            home_team = g.get("h", {}).get("ta")
            away_team = g.get("v", {}).get("ta")

            if not gid or not home_team or not away_team:
                logger.warning(f"Skipping malformed game entry: {g}")
                continue

            box_data = fetch_boxscore(gid, timeout)

            if debug and not debug_logged:
                logger.info("🔍 Boxscore keys:")
                logger.info(json.dumps(box_data.get("g", {}), indent=2))
                debug_logged = True

            odds_key = (normalize_team_name(home_team), normalize_team_name(away_team))
            odds = odds_map.get(odds_key, {})
            home_odds = odds.get("home_odds")
            away_odds = odds.get("away_odds")

            all_rows.extend(
                parse_player_rows(box_data, home_team, away_team, home_odds, away_odds)
            )

        df = pd.DataFrame(all_rows)
        df = ensure_minimum_schema(df, schedule_empty, allow_placeholder)

        archive_new_games()
        NEW_GAMES_FILE.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(NEW_GAMES_FILE, index=False)

        coverage = df["decimal_odds"].notna().mean() * 100
        logger.info(
            f"✅ new_games.csv written → {NEW_GAMES_FILE} "
            f"({len(df)} rows, odds coverage={coverage:.1f}%)"
        )

        return NEW_GAMES_FILE

    except requests.HTTPError as e:
        msg = f"HTTP error while fetching data: {e}"
        logger.error(msg)
        raise DataError(msg)

    except requests.RequestException as e:
        msg = f"Network error: {e}"
        logger.error(msg)
        raise DataError(msg)

    except Exception as e:
        msg = f"Failed to fetch new games: {e}"
        logger.error(msg)
        raise DataError(msg)


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch today's NBA games + player stats + odds.")
    parser.add_argument("--debug", action="store_true", help="Log boxscore structure for first game.")
    parser.add_argument("--allow_placeholder", action="store_true", help="Insert placeholder rows for CI.")
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout.")
    args = parser.parse_args()

    fetch_new_games(debug=args.debug, allow_placeholder=args.allow_placeholder, timeout=args.timeout)
