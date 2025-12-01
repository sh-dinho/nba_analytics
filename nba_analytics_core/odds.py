# nba_analytics_core/odds.py
import requests
import logging
from config import ODDS_API_KEY

BASE_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"

def fetch_odds():
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": "h2h",
        "oddsFormat": "decimal"
    }
    logging.info("Fetching NBA odds from Odds API...")
    resp = requests.get(BASE_URL, params=params)
    resp.raise_for_status()
    data = resp.json()
    logging.info(f"✔ Retrieved odds for {len(data)} games")
    return data