# Path: nba_analytics_core/odds.py

import os
import requests
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ODDS_API_KEY = os.environ.get("ODDS_API_KEY")
BASE_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"

def fetch_odds(home_team=None, away_team=None, region="us", markets="h2h", bookmaker_preference=None):
    """
    Fetch bookmaker odds for NBA H2H market in decimal format.
    Returns dict: {"home_odds": float, "away_odds": float, "bookmaker": str, "last_update": str} or None.
    """
    if not ODDS_API_KEY:
        raise ValueError("ODDS_API_KEY not set. Add it to .env or environment variables.")

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
        logger.error(f"Odds API error: {e}")
        return None

    for game in data:
        home = game.get("home_team")
        away = game.get("away_team")
        if home_team and away_team and home and away:
            if home.lower() == home_team.lower() and away.lower() == away_team.lower():
                bookmakers = game.get("bookmakers", [])
                if not bookmakers:
                    logger.warning(f"No bookmakers available for {home} vs {away}")
                    continue

                # Choose bookmaker
                bookmaker = bookmakers[0]
                if bookmaker_preference:
                    for b in bookmakers:
                        if b["title"].lower() == bookmaker_preference.lower():
                            bookmaker = b
                            break

                markets_data = bookmaker.get("markets", [])
                if not markets_data:
                    logger.warning(f"No markets available for bookmaker {bookmaker['title']}")
                    continue

                outcomes = markets_data[0].get("outcomes", [])
                home_odds = next((o["price"] for o in outcomes if o["name"].lower() == home.lower()), None)
                away_odds = next((o["price"] for o in outcomes if o["name"].lower() == away.lower()), None)

                if home_odds is not None and away_odds is not None:
                    return {
                        "home_odds": float(home_odds),
                        "away_odds": float(away_odds),
                        "bookmaker": bookmaker["title"],
                        "last_update": bookmaker.get("last_update"),
                        "commence_time": game.get("commence_time"),
                        "sport_key": game.get("sport_key")
                    }
                else:
                    logger.warning(f"Odds missing for {home} vs {away} at bookmaker {bookmaker['title']}")

    logger.info(f"No odds found for {home_team} vs {away_team}")
    return None