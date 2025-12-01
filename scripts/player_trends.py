# File: scripts/player_trends.py
import os
import pandas as pd

os.makedirs("results", exist_ok=True)

# Example placeholder data — replace with real pipeline later
data = [
    {"PLAYER_NAME": "LeBron James", "TEAM_ABBREVIATION": "LAL", "points": 28, "rebounds": 8, "assists": 9},
    {"PLAYER_NAME": "Jayson Tatum", "TEAM_ABBREVIATION": "BOS", "points": 26, "rebounds": 7, "assists": 4},
    {"PLAYER_NAME": "Nikola Jokic", "TEAM_ABBREVIATION": "DEN", "points": 25, "rebounds": 12, "assists": 10},
]
df = pd.DataFrame(data)
df.to_csv("results/player_trends.csv", index=False)

print("✅ Player trends saved to results/player_trends.csv")