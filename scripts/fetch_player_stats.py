# File: scripts/fetch_player_stats.py
import pandas as pd
import requests

def main():
    # Example: NBA.com stats endpoint (simplified)
    url = "https://stats.nba.com/stats/leaguedashplayerstats"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://www.nba.com/stats"
    }
    params = {
        "Season": "2025-26",
        "SeasonType": "Regular Season",
        "PerMode": "PerGame"
    }

    r = requests.get(url, headers=headers, params=params)
    r.raise_for_status()
    data = r.json()

    # Parse into DataFrame
    headers = data["resultSets"][0]["headers"]
    rows = data["resultSets"][0]["rowSet"]
    df = pd.DataFrame(rows, columns=headers)

    # Keep only relevant columns
    df = df[["PLAYER_NAME", "TEAM_ABBREVIATION", "GP", "PTS", "REB", "AST"]]
    df.rename(columns={
        "PLAYER_NAME": "player",
        "TEAM_ABBREVIATION": "team",
        "PTS": "points",
        "REB": "rebounds",
        "AST": "assists"
    }, inplace=True)

    # Save to CSV
    df.to_csv("data/player_stats.csv", index=False)
    print("✅ Live player stats saved to data/player_stats.csv")

if __name__ == "__main__":
    main()