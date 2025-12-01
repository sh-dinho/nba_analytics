# core/odds.py (Updated)
import os
import logging
import requests
import pandas as pd
from nba_analytics_core.utils import get_standardized_team_name # New standardization utility
from config import ODDS_API_KEY # New: Use config for key placeholder

def american_to_decimal(american_odds: int) -> float:
    if american_odds > 0:
        return 1 + (american_odds / 100.0)
    return 1 + (100.0 / abs(american_odds))

def fetch_game_odds(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fetch odds for today's games, standardize team names, and return a DataFrame.
    """
    # Use config/ENV for key
    api_key = ODDS_API_KEY
    base_url = os.getenv("ODDS_BASE_URL", "https://api.the-odds-api.com/v4/sports/basketball_nba/odds")
    market = os.getenv("ODDS_MARKET", "h2h")
    bookmaker = os.getenv("ODDS_BOOKMAKER", "betonline")

    if api_key == "YOUR_PLACEHOLDER_KEY" or not api_key:
        logging.warning("No valid ODDS_API_KEY set; falling back to placeholder odds.")
        out = games_df[["game_id", "home_team", "away_team"]].copy()
        out["home_decimal_odds"] = american_to_decimal(-110)
        out["away_decimal_odds"] = american_to_decimal(-110)
        return out

    params = {
        "apiKey": api_key,
        "markets": market,
        "bookmakers": bookmaker,
        "oddsFormat": "american",
        "dateFormat": "iso"
    }
    
    # ... (Error handling and API request logic remains the same) ...
    # (Simplified for display, assuming full error handling is kept)

    try:
        resp = requests.get(base_url, params=params, timeout=15)
        if resp.status_code != 200:
             # Graceful fallback to placeholders
            logging.error(f"Odds API error {resp.status_code}: {resp.text}")
            out = games_df[["game_id", "home_team", "away_team"]].copy()
            out["home_decimal_odds"] = american_to_decimal(-110)
            out["away_decimal_odds"] = american_to_decimal(-110)
            return out
        data = resp.json()
    except Exception as e:
        # Graceful fallback
        logging.error(f"Failed to fetch odds: {e}")
        out = games_df[["game_id", "home_team", "away_team"]].copy()
        out["home_decimal_odds"] = american_to_decimal(-110)
        out["away_decimal_odds"] = american_to_decimal(-110)
        return out


    # Build a simple odds table using team standardization
    odds_records = []
    for item in data:
        try:
            # ... (Bookmaker and market parsing logic remains the same) ...
            bkm = next((b for b in item.get("bookmakers", []) if b.get("key") == bookmaker), None)
            if not bkm: continue
            mkt = next((m for m in bkm.get("markets", []) if m.get("key") == market), None)
            if not mkt: continue

            outcomes = {o["name"]: o["price"] for o in mkt.get("outcomes", [])}
            
            home_name_api = item.get("home_team")
            away_name_api = item.get("away_team")
            if not home_name_api or not away_name_api: continue

            home_odds = outcomes.get(home_name_api)
            away_odds = outcomes.get(away_name_api)
            if home_odds is None or away_odds is None: continue

            # --- IMPROVEMENT: Standardize Team Names ---
            standard_home = get_standardized_team_name(home_name_api)
            standard_away = get_standardized_team_name(away_name_api)

            odds_records.append({
                "home_team": standard_home, # Standardized name
                "away_team": standard_away, # Standardized name
                "home_decimal_odds": american_to_decimal(int(home_odds)),
                "away_decimal_odds": american_to_decimal(int(away_odds)),
            })
        except Exception:
            continue

    odds_df = pd.DataFrame(odds_records)
    # ... (Rest of fallback and merging logic remains the same) ...
    if odds_df.empty:
        logging.warning("Odds API returned no matchable records; falling back to placeholders.")
        out = games_df[["game_id", "home_team", "away_team"]].copy()
        out["home_decimal_odds"] = american_to_decimal(-110)
        out["away_decimal_odds"] = american_to_decimal(-110)
        return out

    # Merge by standardized team names
    merged = games_df.merge(
        odds_df.drop_duplicates(subset=["home_team", "away_team"]),
        on=["home_team", "away_team"],
        how="left"
    )

    merged["home_decimal_odds"] = merged["home_decimal_odds"].fillna(american_to_decimal(-110))
    merged["away_decimal_odds"] = merged["away_decimal_odds"].fillna(american_to_decimal(-110))

    return merged[["game_id", "home_team", "away_team", "home_decimal_odds", "away_decimal_odds"]]