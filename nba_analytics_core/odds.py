import os
import requests

ODDS_API_KEY = os.environ.get("ODDS_API_KEY")
BASE_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"

def fetch_odds(home_team=None, away_team=None, region="us", markets="h2h"):
    """
    Fetch bookmaker odds for NBA H2H market in decimal format.
    Returns dict: {"home_odds": float, "away_odds": float} or None.
    """
    if not ODDS_API_KEY:
        raise ValueError("ODDS_API_KEY not set. Add it to .env or Streamlit secrets.")

    params = {
        "apiKey": ODDS_API_KEY,
        "regions": region,
        "markets": markets,
        "oddsFormat": "decimal"
    }
    resp = requests.get(BASE_URL, params=params, timeout=15)
    if resp.status_code != 200:
        print(f"Odds API error: {resp.status_code} {resp.text}")
        return None

    data = resp.json()
    for game in data:
        home = game.get("home_team")
        away = game.get("away_team")
        if home_team and away_team and home and away:
            if home.lower() == home_team.lower() and away.lower() == away_team.lower():
                # choose first bookmaker available
                if not game["bookmakers"]:
                    continue
                market = game["bookmakers"][0]["markets"][0]["outcomes"]
                home_odds = next((o["price"] for o in market if o["name"].lower() == home.lower()), None)
                away_odds = next((o["price"] for o in market if o["name"].lower() == away.lower()), None)
                if home_odds and away_odds:
                    return {"home_odds": float(home_odds), "away_odds": float(away_odds)}
    return None