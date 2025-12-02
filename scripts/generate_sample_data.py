import os
import pandas as pd
from nba_api.stats.endpoints import scoreboardv2
import requests

ODDS_API_KEY = "YOUR_THEODDSAPI_KEY"  # replace with your key

def fetch_today_games():
    try:
        sb = scoreboardv2.ScoreboardV2()
        games = sb.game_header.get_data_frame()
        lineups = sb.line_score.get_data_frame()
    except Exception as e:
        print(f"Error fetching today's games: {e}")
        return []

    teams_map = {row["TEAM_ID"]: row["TEAM_ABBREVIATION"] for _, row in lineups.iterrows()}

    today_games = []
    for _, row in games.iterrows():
        home_team = teams_map.get(row["HOME_TEAM_ID"])
        away_team = teams_map.get(row["VISITOR_TEAM_ID"])
        if home_team and away_team:
            today_games.append({
                "game_id": row["GAME_ID"],
                "date": row["GAME_DATE_EST"],
                "home_team": home_team,
                "away_team": away_team
            })
    return today_games


def fetch_odds():
    try:
        resp = requests.get(
            "https://api.the-odds-api.com/v4/sports/basketball_nba/odds",
            params={"regions": "us", "markets": "h2h,spreads,totals", "apiKey": ODDS_API_KEY}
        )
        if resp.ok:
            return resp.json()
    except Exception as e:
        print(f"Odds fetch failed: {e}")
    return []


def fetch_injuries():
    try:
        resp = requests.get("https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries")
        if resp.ok:
            return resp.json().get("injuries", [])
    except Exception as e:
        print(f"Injury fetch failed: {e}")
    return []


def main():
    os.makedirs("data", exist_ok=True)

    games = fetch_today_games()
    odds_data = fetch_odds()
    injuries_data = fetch_injuries()

    # --- Save odds ---
    odds_rows = []
    for g in games:
        # Simplify: attach first odds entry matching home/away
        if odds_data:
            odds_rows.append({
                "game_id": g["game_id"],
                "home_team": g["home_team"],
                "away_team": g["away_team"],
                "home_moneyline": odds_data[0]["bookmakers"][0]["markets"][0]["outcomes"][0]["price"],
                "away_moneyline": odds_data[0]["bookmakers"][0]["markets"][0]["outcomes"][1]["price"],
                "spread": odds_data[0]["bookmakers"][0]["markets"][1]["outcomes"][0]["point"],
                "total": odds_data[0]["bookmakers"][0]["markets"][2]["outcomes"][0]["point"]
            })
    pd.DataFrame(odds_rows).to_csv("data/odds.csv", index=False)
    print("✅ data/odds.csv saved")

    # --- Save injuries ---
    inj_rows = []
    for inj in injuries_data:
        team = inj["team"]["abbreviation"]
        for player in inj.get("players", []):
            inj_rows.append({
                "date": pd.Timestamp.today().strftime("%Y-%m-%d"),
                "team": team,
                "player": player["fullName"],
                "status": player.get("status"),
                "impact_minutes": 30  # placeholder, you can refine
            })
    pd.DataFrame(inj_rows).to_csv("data/injuries.csv", index=False)
    print("✅ data/injuries.csv saved")


if __name__ == "__main__":
    main()