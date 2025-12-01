# File: nba_analytics_core/odds.py

import os
import requests

ODDS_API_KEY = os.environ.get("ODDS_API_KEY")
BASE_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"

def fetch_odds(home_team=None, away_team=None, region="us", markets="h2h", bookmaker_preference=None):
    """
    Fetch bookmaker odds for NBA H2H market in decimal format.
    Returns dict: {"home_odds": float, "away_odds": float, "bookmaker": str, "last_update": str} or None.
    """
    if not ODDS_API_KEY:
        raise ValueError("ODDS_API_KEY not set. Add it to .env or Streamlit secrets.")

    params = {
        "apiKey": ODDS_API_KEY,
        "regions": region,
        "markets": markets,
        "oddsFormat": "decimal"
    }

    try:
        resp = requests.get(BASE_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        print(f"Odds API error: {e}")
        return None

    for game in data:
        home = game.get("home_team")
        away = game.get("away_team")
        if home_team and away_team and home and away:
            if home.lower() == home_team.lower() and away.lower() == away_team.lower():
                if not game["bookmakers"]:
                    continue

                # Choose bookmaker
                bookmaker = game["bookmakers"][0]
                if bookmaker_preference:
                    for b in game["bookmakers"]:
                        if b["title"].lower() == bookmaker_preference.lower():
                            bookmaker = b
                            break

                market = bookmaker["markets"][0]["outcomes"]
                home_odds = next((o["price"] for o in market if o["name"].lower() == home.lower()), None)
                away_odds = next((o["price"] for o in market if o["name"].lower() == away.lower()), None)

                if home_odds and away_odds:
                    return {
                        "home_odds": float(home_odds),
                        "away_odds": float(away_odds),
                        "bookmaker": bookmaker["title"],
                        "last_update": bookmaker.get("last_update")
                    }
    return None