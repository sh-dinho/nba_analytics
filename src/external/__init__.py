# ============================================================
# File: src/external/odds.py
# ============================================================

import requests
import pandas as pd


def fetch_espn_odds():
    """
    Fetch NBA odds from ESPN public API.
    Returns:
        game_id, home_team, away_team, home_ml, away_ml
    """
    url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
    resp = requests.get(url).json()

    rows = []
    for event in resp.get("events", []):
        comp = event.get("competitions", [{}])[0]
        odds_list = comp.get("odds", [])
        odds = odds_list[0] if odds_list else {}

        competitors = comp.get("competitors", [])
        if len(competitors) < 2:
            continue

        home = [c for c in competitors if c.get("homeAway") == "home"]
        away = [c for c in competitors if c.get("homeAway") == "away"]
        if not home or not away:
            continue

        home = home[0]
        away = away[0]

        rows.append(
            {
                "game_id": event["id"],
                "home_team": home["team"]["displayName"],
                "away_team": away["team"]["displayName"],
                "home_ml": odds.get("homeMoneyLine"),
                "away_ml": odds.get("awayMoneyLine"),
            }
        )

    return pd.DataFrame(rows)
