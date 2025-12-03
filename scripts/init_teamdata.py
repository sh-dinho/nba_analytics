import sqlite3
import pandas as pd
import os

# Path to your DB file
db_path = r"C:\Users\Mohamadou\projects\nba_analytics\Data\TeamData.sqlite"

# Ensure Data folder exists
os.makedirs(os.path.dirname(db_path), exist_ok=True)

# Connect (creates file if it doesn't exist)
con = sqlite3.connect(db_path)

# Example data for multiple seasons
season_data = {
    "2023_2024": {
        "TEAM_ABBREVIATION": ["BOS", "LAL", "MIA"],
        "WINS": [50, 45, 42],
        "LOSSES": [32, 37, 40],
        "POINTS_FOR": [8900, 8700, 8600],
        "POINTS_AGAINST": [8700, 8800, 8650],
    },
    "2024_2025": {
        "TEAM_ABBREVIATION": ["BOS", "LAL", "MIA"],
        "WINS": [55, 48, 44],
        "LOSSES": [27, 34, 38],
        "POINTS_FOR": [9100, 8850, 8700],
        "POINTS_AGAINST": [8800, 8900, 8750],
    },
    "2025_2026": {
        "TEAM_ABBREVIATION": ["BOS", "LAL", "MIA"],
        "WINS": [15, 12, 10],   # sample partial season
        "LOSSES": [5, 8, 9],
        "POINTS_FOR": [2200, 2100, 2000],
        "POINTS_AGAINST": [2000, 2050, 1980],
    }
}

# Write each season table
for season, data in season_data.items():
    df = pd.DataFrame(data)
    table_name = f"teamdata_{season}"
    df.to_sql(table_name, con, if_exists="replace", index=False)
    print(f"✅ Created table {table_name} with {len(df)} rows")

con.close()
print(f"🎉 Database initialized at {db_path}")