# File: scripts/player_trends.py
import os
import pandas as pd

os.makedirs("results", exist_ok=True)

def generate_player_trends():
    # Example: replace with your actual data source
    data = [
        {"week": "2025-11-17", "PLAYER_NAME": "LeBron James", "TEAM_ABBREVIATION": "LAL", "points": 25, "rebounds": 8, "assists": 9, "TS_PCT": 0.58},
        {"week": "2025-11-24", "PLAYER_NAME": "LeBron James", "TEAM_ABBREVIATION": "LAL", "points": 28, "rebounds": 7, "assists": 10, "TS_PCT": 0.61},
        {"week": "2025-11-17", "PLAYER_NAME": "Jayson Tatum", "TEAM_ABBREVIATION": "BOS", "points": 26, "rebounds": 7, "assists": 4, "TS_PCT": 0.55},
        {"week": "2025-11-24", "PLAYER_NAME": "Jayson Tatum", "TEAM_ABBREVIATION": "BOS", "points": 27, "rebounds": 8, "assists": 5, "TS_PCT": 0.57},
    ]
    df = pd.DataFrame(data)

    # Sort by player and week to compute changes
    df = df.sort_values(["PLAYER_NAME", "week"])

    # Compute deltas week-over-week
    df["PTS_change"] = df.groupby("PLAYER_NAME")["points"].diff()
    df["REB_change"] = df.groupby("PLAYER_NAME")["rebounds"].diff()
    df["AST_change"] = df.groupby("PLAYER_NAME")["assists"].diff()
    df["TS_PCT_change"] = df.groupby("PLAYER_NAME")["TS_PCT"].diff()

    df.to_csv("results/player_trends.csv", index=False)
    print("✅ Player trends saved to results/player_trends.csv")

if __name__ == "__main__":
    generate_player_trends()