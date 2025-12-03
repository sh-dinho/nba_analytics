# ============================================================
# File: scripts/fetch_new_games.py
# Purpose: Fetch today's NBA games with player stats + odds
# ============================================================

import os
import argparse
import pandas as pd
import datetime
import requests
import json
from pathlib import Path
from dotenv import load_dotenv
from core.config import NEW_GAMES_FILE, BASE_DATA_DIR
from core.log_config import setup_logger
from core.exceptions import DataError

logger = setup_logger("fetch_new_games")

REQUIRED_COLUMNS = [
    "PLAYER_NAME", "TEAM_ABBREVIATION", "TEAM_HOME", "TEAM_AWAY",
    "PTS", "AST", "REB", "GAMES_PLAYED", "decimal_odds"
]

NBA_SCHEDULE_URL = "https://data.nba.com/data/v2015/json/mobile_teams/nba/2025/scores/00_todays_scores.json"
BOX_URL_TEMPLATE = "https://data.nba.com/data/v2015/json/mobile_teams/nba/2025/scores/gamedetail/{gid}_gamedetail.json"

# Odds API setup
load_dotenv()
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"


def debug_boxscore_structure(box_data: dict):
    """Log the JSON structure of the first game for debugging."""
    logger.info("🔍 Debugging boxscore JSON structure...")
    logger.info(f"Top-level keys: {list(box_data.keys())}")
    if "g" in box_data:
        logger.info(f"Keys under 'g': {list(box_data['g'].keys())}")
        if "pstsg" in box_data["g"]:
            players = box_data["g"]["pstsg"]
            if players:
                logger.info("Sample player entry:")
                logger.info(json.dumps(players[0], indent=2))


def fetch_odds() -> dict:
    """Fetch odds for today's NBA games from OddsAPI."""
    if not ODDS_API_KEY:
        logger.warning("⚠️ No ODDS_API_KEY found in .env")
        return {}

    try:
        params = {
            "apiKey": ODDS_API_KEY,
            "regions": "us",
            "markets": "h2h",
            "oddsFormat": "decimal",
            "dateFormat": "iso"
        }
        response = requests.get(ODDS_API_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        odds_map = {}
        for game in data:
            home_team = game["home_team"]
            away_team = game["away_team"]
            home_odds, away_odds = [], []
            for book in game.get("bookmakers", []):
                for market in book.get("markets", []):
                    for outcome in market.get("outcomes", []):
                        if outcome["name"] == home_team:
                            home_odds.append(outcome["price"])
                        elif outcome["name"] == away_team:
                            away_odds.append(outcome["price"])
            odds_map[(home_team, away_team)] = {
                "home_odds": sum(home_odds) / len(home_odds) if home_odds else None,
                "away_odds": sum(away_odds) / len(away_odds) if away_odds else None,
            }
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


def fetch_new_games(debug: bool = False, allow_placeholder: bool = False) -> str:
    """Fetch today's games with player stats and merge odds."""
    os.makedirs(BASE_DATA_DIR, exist_ok=True)

    try:
        today = datetime.date.today()
        logger.info(f"📡 Fetching today's NBA games for {today}...")

        # Get today's scoreboard
        response = requests.get(NBA_SCHEDULE_URL, timeout=30)
        response.raise_for_status()
        data = response.json()

        games_today = [g for g in data.get("gs", {}).get("g", []) if g.get("gdte") == today.strftime("%Y-%m-%d")]
        schedule_empty = len(games_today) == 0

        odds_map = fetch_odds()
        all_players = []
        first_game_logged = False

        for g in games_today:
            gid = g.get("gid")
            home_team = g.get("h", {}).get("ta")
            away_team = g.get("v", {}).get("ta")

            # Fetch boxscore for this game
            box_url = BOX_URL_TEMPLATE.format(gid=gid)
            box_resp = requests.get(box_url, timeout=30)
            box_resp.raise_for_status()
            box_data = box_resp.json()

            # Debug log structure for the first game only if flag enabled
            if debug and not first_game_logged:
                debug_boxscore_structure(box_data)
                first_game_logged = True

            # Odds lookup
            odds_entry = odds_map.get((home_team, away_team), {})
            home_odds = odds_entry.get("home_odds")
            away_odds = odds_entry.get("away_odds")

            # Player stats
            players = box_data.get("g", {}).get("pstsg", [])
            if not players:
                # No stats yet, create placeholder rows
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
                    decimal_odds = home_odds if team_abbr == home_team else away_odds
                    all_players.append({
                        "PLAYER_NAME": f"{p.get('fn','')} {p.get('ln','')}".strip(),
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
        df = ensure_minimum_schema(df, allow_placeholder, schedule_empty)

        df.to_csv(NEW_GAMES_FILE, index=False)
        logger.info(f"✅ new_games.csv saved to {NEW_GAMES_FILE} with {len(df)} rows")
        return NEW_GAMES_FILE

    except Exception as e:
        logger.error(f"❌ Failed to fetch new games: {e}")
        raise DataError(f"Failed to fetch new games: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch today's games")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging of JSON structure")
    parser.add_argument("--allow_placeholder", action="store_true", help="Insert NO_GAMES_TODAY or NO_STATS_YET row if feed is empty")
    args = parser.parse_args()
    fetch_new_games(debug=args.debug, allow_placeholder=args.allow_placeholder)