# core/odds.py
import os
import logging
import requests
import pandas as pd

def american_to_decimal(american_odds: int) -> float:
    if american_odds > 0:
        return 1 + (american_odds / 100.0)
    return 1 + (100.0 / abs(american_odds))

def fetch_game_odds(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fetch odds for today's games and return a DataFrame with columns:
    game_id, home_team, away_team, home_decimal_odds, away_decimal_odds
    """
    api_key = os.getenv("ODDS_API_KEY")
    base_url = os.getenv("ODDS_BASE_URL", "https://api.the-odds-api.com/v4/sports/basketball_nba/odds")
    market = os.getenv("ODDS_MARKET", "h2h")
    bookmaker = os.getenv("ODDS_BOOKMAKER", "betonline")

    if not api_key:
        logging.warning("No ODDS_API_KEY set; falling back to placeholder odds.")
        # Fallback: -110 both sides
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

    try:
        resp = requests.get(base_url, params=params, timeout=15)
        if resp.status_code != 200:
            logging.error(f"Odds API error {resp.status_code}: {resp.text}")
            # Graceful fallback to placeholders
            out = games_df[["game_id", "home_team", "away_team"]].copy()
            out["home_decimal_odds"] = american_to_decimal(-110)
            out["away_decimal_odds"] = american_to_decimal(-110)
            return out

        data = resp.json()
    except Exception as e:
        logging.error(f"Failed to fetch odds: {e}")
        # Graceful fallback
        out = games_df[["game_id", "home_team", "away_team"]].copy()
        out["home_decimal_odds"] = american_to_decimal(-110)
        out["away_decimal_odds"] = american_to_decimal(-110)
        return out

    # Normalize and match to today's games by team names
    rows = []
    for item in data:
        # item structure typically includes 'bookmakers' -> 'markets' -> 'outcomes'
        try:
            bkm = next((b for b in item.get("bookmakers", []) if b.get("key") == bookmaker), None)
            if not bkm:
                continue
            mkt = next((m for m in bkm.get("markets", []) if m.get("key") == market), None)
            if not mkt:
                continue

            outcomes = mkt.get("outcomes", [])
            # Try to identify home vs away by names
            # Outcomes have 'name' (team) and 'price' (american odds)
            # We'll map to the input games_df rows
            rows.append({
                "home_team": None,  # to be aligned later
                "away_team": None,
                "home_american_odds": None,
                "away_american_odds": None,
                "event_id": item.get("id")
            })
            # We'll fill by matching below
        except Exception:
            continue

    # Build a simple odds table by matching teams
    # Note: Different APIs vary. This example expects exact team name matches.
    # You may need a team alias map for inconsistencies.
    odds_records = []
    for item in data:
        try:
            bkm = next((b for b in item.get("bookmakers", []) if b.get("key") == bookmaker), None)
            if not bkm:
                continue
            mkt = next((m for m in bkm.get("markets", []) if m.get("key") == market), None)
            if not mkt:
                continue

            outcomes = {o["name"]: o["price"] for o in mkt.get("outcomes", [])}
            # Attempt match to games_df by away/home team names
            # Some APIs include 'home_team' and 'away_team' fields directly:
            home_name = item.get("home_team")
            away_name = item.get("away_team")
            if not home_name or not away_name:
                # fallback: try to infer from games_df entries if team names in outcomes
                continue

            home_odds = outcomes.get(home_name)
            away_odds = outcomes.get(away_name)
            if home_odds is None or away_odds is None:
                continue

            odds_records.append({
                "home_team": home_name,
                "away_team": away_name,
                "home_decimal_odds": american_to_decimal(int(home_odds)),
                "away_decimal_odds": american_to_decimal(int(away_odds)),
            })
        except Exception:
            continue

    odds_df = pd.DataFrame(odds_records)
    if odds_df.empty:
        logging.warning("Odds API returned no matchable records; falling back to placeholders.")
        out = games_df[["game_id", "home_team", "away_team"]].copy()
        out["home_decimal_odds"] = american_to_decimal(-110)
        out["away_decimal_odds"] = american_to_decimal(-110)
        return out

    # Merge odds by team names; if duplicates, pick first
    merged = games_df.merge(
        odds_df.drop_duplicates(subset=["home_team", "away_team"]),
        on=["home_team", "away_team"],
        how="left"
    )

    # Fill any missing with placeholder odds
    merged["home_decimal_odds"] = merged["home_decimal_odds"].fillna(american_to_decimal(-110))
    merged["away_decimal_odds"] = merged["away_decimal_odds"].fillna(american_to_decimal(-110))

    return merged[["game_id", "home_team", "away_team", "home_decimal_odds", "away_decimal_odds"]]