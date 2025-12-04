# ============================================================
# File: scripts/fetch_new_games.py
# Purpose: Fetch today's NBA games with player stats + odds
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

# Required output schema
REQUIRED_COLUMNS = [
    "PLAYER_NAME", "TEAM_ABBREVIATION", "TEAM_HOME", "TEAM_AWAY",
    "PTS", "AST", "REB", "GAMES_PLAYED", "decimal_odds"
]

# NBA mobile JSON feeds (season component should be updated yearly)
NBA_SEASON_YEAR = "2025"  # adjust when season changes
NBA_SCHEDULE_URL = f"https://data.nba.com/data/v2015/json/mobile_teams/nba/{NBA_SEASON_YEAR}/scores/00_todays_scores.json"
BOX_URL_TEMPLATE = f"https://data.nba.com/data/v2015/json/mobile_teams/nba/{NBA_SEASON_YEAR}/scores/gamedetail/{{gid}}_gamedetail.json"

# Odds API setup
load_dotenv()
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"


def archive_new_games():
    """Archive existing new_games file before overwriting."""
    if NEW_GAMES_FILE.exists():
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_file = ARCHIVE_DIR / f"new_games_{ts}.csv"
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy(NEW_GAMES_FILE, archive_file)
        logger.info(f"📦 Archived new games to {archive_file}")


def debug_boxscore_structure(box_data: dict):
    """Log the JSON structure of the first game for debugging."""
    logger.info("🔍 Debugging boxscore JSON structure...")
    try:
        logger.info(f"Top-level keys: {list(box_data.keys())}")
        g = box_data.get("g", {})
        logger.info(f"Keys under 'g': {list(g.keys())}")
        players = g.get("pstsg", [])
        if players:
            logger.info("Sample player entry:")
            logger.info(json.dumps(players[0], indent=2))
    except Exception:
        # Defensive: avoid breaking on odd JSON structures
        logger.warning("Could not fully log boxscore structure.")


def fetch_odds(timeout: int = 30) -> dict:
    """Fetch odds for today's NBA games from Odds API. Returns a dict keyed by (home_team, away_team)."""
    if not ODDS_API_KEY:
        logger.warning("⚠️ No ODDS_API_KEY found in environment")
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
            home_team = game.get("home_team")
            away_team = game.get("away_team")
            if not home_team or not away_team:
                continue

            home_prices, away_prices = [], []
            for book in game.get("bookmakers", []):
                for market in book.get("markets", []):
                    for outcome in market.get("outcomes", []):
                        name = outcome.get("name")
                        price = outcome.get("price")
                        if name == home_team and price is not None:
                            home_prices.append(price)
                        elif name == away_team and price is not None:
                            away_prices.append(price)

            odds_map[(home_team, away_team)] = {
                "home_odds": (sum(home_prices) / len(home_prices)) if home_prices else None,
                "away_odds": (sum(away_prices) / len(away_prices)) if away_prices else None,
            }
        logger.info(f"🎲 Fetched odds for {len(odds_map)} games")
        return odds_map
    except Exception as e:
        logger.warning(f"⚠️ Failed to fetch odds: {e}")
        return {}


def ensure_minimum_schema(df: pd.DataFrame, allow_placeholder: bool, schedule_empty: bool) -> pd.DataFrame:
    """
    Guarantee that the DataFrame has all required columns.
    If missing, add them with None values.
    If completely empty and allow_placeholder=True, insert a placeholder row.
    Distinguishes between NO_GAMES_TODAY vs NO_STATS_YET.
    """
    if df.empty and allow_placeholder:
        placeholder_type = "NO_GAMES_TODAY" if schedule_empty else "NO_STATS_YET"
        df = pd.DataFrame([{
            "PLAYER_NAME": placeholder_type,
            "TEAM_ABBREVIATION": None,
            "TEAM_HOME": None,
            "TEAM_AWAY": None,
            "PTS": None,
            "AST": None,
            "REB": None,
            "GAMES_PLAYED": None,
            "decimal_odds": None,
        }])

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = None

    return df[REQUIRED_COLUMNS]


def fetch_new_games(debug: bool = False, allow_placeholder: bool = False, timeout: int = 30) -> Path:
    """Fetch today's games with player stats and merge odds; persist CSV and return its path."""
    ensure_dirs(strict=False)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        today = datetime.date.today()
        logger.info(f"📡 Fetching today's NBA games for {today}...")

        # Get today's scoreboard
        resp = requests.get(NBA_SCHEDULE_URL, timeout=timeout)
        resp.raise_for_status()
        scoreboard = resp.json()

        games_node = scoreboard.get("gs", {}).get("g", [])
        # Handle if "g" is a dict or list
        if isinstance(games_node, dict):
            games_iterable = [games_node]
        else:
            games_iterable = games_node

        # Filter today's games (gdte in ISO format, e.g., '2025-12-03')
        games_today = [g for g in games_iterable if g.get("gdte") == today.strftime("%Y-%m-%d")]
        schedule_empty = len(games_today) == 0

        odds_map = fetch_odds(timeout=timeout)
        all_players = []
        first_game_logged = False

        for g in games_today:
            gid = g.get("gid")
            home_team = g.get("h", {}).get("ta")
            away_team = g.get("v", {}).get("ta")
            if not gid or not home_team or not away_team:
                logger.warning(f"Skipping malformed game entry: gid={gid}, home={home_team}, away={away_team}")
                continue

            # Fetch boxscore for this game
            box_url = BOX_URL_TEMPLATE.format(gid=gid)
            box_resp = requests.get(box_url, timeout=timeout)
            box_resp.raise_for_status()
            box_data = box_resp.json()

            # Debug first game structure if enabled
            if debug and not first_game_logged:
                debug_boxscore_structure(box_data)
                first_game_logged = True

            # Odds lookup
            odds_entry = odds_map.get((home_team, away_team), {})
            home_odds = odds_entry.get("home_odds")
            away_odds = odds_entry.get("away_odds")

            # Player stats under g.pstsg
            players = box_data.get("g", {}).get("pstsg", [])
            if not players:
                # No stats yet; write team-level placeholders
                all_players.append({
                    "PLAYER_NAME": None,
                    "TEAM_ABBREVIATION": home_team,
                    "TEAM_HOME": home_team,
                    "TEAM_AWAY": away_team,
                    "PTS": None,
                    "AST": None,
                    "REB": None,
                    "GAMES_PLAYED": None,
                    "decimal_odds": home_odds,
                })
                all_players.append({
                    "PLAYER_NAME": None,
                    "TEAM_ABBREVIATION": away_team,
                    "TEAM_HOME": home_team,
                    "TEAM_AWAY": away_team,
                    "PTS": None,
                    "AST": None,
                    "REB": None,
                    "GAMES_PLAYED": None,
                    "decimal_odds": away_odds,
                })
            else:
                for p in players:
                    team_abbr = p.get("ta")
                    first_name = p.get("fn", "")
                    last_name = p.get("ln", "")
                    player_name = f"{first_name} {last_name}".strip()
                    decimal_odds = home_odds if team_abbr == home_team else away_odds
                    all_players.append({
                        "PLAYER_NAME": player_name or None,
                        "TEAM_ABBREVIATION": team_abbr,
                        "TEAM_HOME": home_team,
                        "TEAM_AWAY": away_team,
                        "PTS": p.get("pts"),
                        "AST": p.get("ast"),
                        "REB": p.get("reb"),
                        "GAMES_PLAYED": p.get("gp"),
                        "decimal_odds": decimal_odds,
                    })

        df = pd.DataFrame(all_players)
        df = ensure_minimum_schema(df, allow_placeholder=allow_placeholder, schedule_empty=schedule_empty)

        # Persist with archiving
        archive_new_games()
        NEW_GAMES_FILE.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(NEW_GAMES_FILE, index=False)
        logger.info(f"✅ new_games.csv saved to {NEW_GAMES_FILE} with {len(df)} rows")

        return NEW_GAMES_FILE

    except requests.HTTPError as e:
        msg = f"HTTP error while fetching games: {e}"
        logger.error(f"❌ {msg}")
        raise DataError(msg)
    except requests.RequestException as e:
        msg = f"Network error while fetching games: {e}"
        logger.error(f"❌ {msg}")
        raise DataError(msg)
    except FileError as e:
        msg = f"File operation error: {e}"
        logger.error(f"❌ {msg}")
        raise
    except Exception as e:
        msg = f"Failed to fetch new games: {e}"
        logger.error(f"❌ {msg}")
        raise DataError(msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch today's games with player stats + odds")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging of JSON structure")
    parser.add_argument("--allow_placeholder", action="store_true", help="Insert NO_GAMES_TODAY or NO_STATS_YET row if feed is empty")
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout in seconds")
    args = parser.parse_args()

    fetch_new_games(debug=args.debug, allow_placeholder=args.allow_placeholder, timeout=args.timeout)
