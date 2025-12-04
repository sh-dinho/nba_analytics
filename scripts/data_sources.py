# ============================================================
# File: scripts/data_sources.py
# Purpose: NBA stats, schedule, and odds using config.toml and env secrets
# ============================================================

import os
from datetime import datetime
import requests
from nba_api.stats.endpoints import LeagueGameFinder, leaguedashteamstats
from dotenv import load_dotenv
from core.config_loader import ConfigLoader

# -------------------------------
# Load secrets from .env
# -------------------------------
load_dotenv()

THE_ODDS_API_KEY = os.getenv("THE_ODDS_API_KEY")
APIFY_ODDS_API_KEY = os.getenv("APIFY_ODDS_API_KEY")

if not THE_ODDS_API_KEY:
    raise ValueError("❌ THE_ODDS_API_KEY not found in environment")
if not APIFY_ODDS_API_KEY:
    print("⚠️ APIFY_ODDS_API_KEY not set, fallback odds may not work.")

# -------------------------------
# Load configuration
# -------------------------------
config_loader = ConfigLoader("config.toml")

def get_season_data(section: str, season_label: str):
    """Return validated season block from config.toml"""
    season_data = config_loader.get_season(section, season_label)
    if not config_loader.validate_season(season_data):
        raise ValueError(f"❌ Season data for {season_label} failed validation")
    return season_data

# -------------------------------
# NBA API Helpers
# -------------------------------
def get_games_for_season(season_label: str):
    season_data = get_season_data("get-data", season_label)
    start_date = season_data["start_date"]
    end_date = season_data["end_date"]
    df = LeagueGameFinder(date_from=start_date, date_to=end_date).get_data_frames()[0]
    return df

def get_team_stats_for_season(season_label: str):
    df = leaguedashteamstats.LeagueDashTeamStats(season=season_label).get_data_frames()[0]
    return df

# -------------------------------
# Odds API Helpers
# -------------------------------
def get_odds_the_odds_api():
    url = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds/"
    params = {
        "regions": "us",
        "markets": "h2h,totals,spreads",
        "apiKey": THE_ODDS_API_KEY,
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            raise ValueError("No odds returned from The Odds API")
        return data
    except Exception as e:
        print(f"❌ The Odds API failed: {e}")
        return None

def get_odds_apify():
    if not APIFY_ODDS_API_KEY:
        return None
    url = f"https://api.apify.com/v2/acts/odds-api~nba-odds/runs/last/dataset/items?token={APIFY_ODDS_API_KEY}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            raise ValueError("No odds returned from Apify")
        return data
    except Exception as e:
        print(f"❌ Apify Odds API failed: {e}")
        return None

def get_odds():
    odds = get_odds_the_odds_api()
    if odds:
        return odds
    print("⚠️ Falling back to Apify odds API...")
    return get_odds_apify()

# -------------------------------
# Build data URL (if needed)
# -------------------------------
def build_data_url(season_label: str):
    season_data = get_season_data("get-data", season_label)
    return config_loader.build_data_url(season_data)

# -------------------------------
# CLI Testing
# -------------------------------
if __name__ == "__main__":
    test_season = "2023-24"

    print(f"📅 Fetching NBA games for season {test_season}...")
    games = get_games_for_season(test_season)
    print(f"✅ {len(games)} games found.")

    print(f"📊 Fetching team stats for season {test_season}...")
    stats = get_team_stats_for_season(test_season)
    print(f"✅ {len(stats)} team stats rows found.")

    print(f"🎲 Fetching NBA odds...")
    odds_data = get_odds()
    if odds_data:
        print(f"✅ {len(odds_data)} odds entries found.")
    else:
        print("❌ No odds data available.")

    print(f"🔗 Built API URL: {build_data_url(test_season)}")
