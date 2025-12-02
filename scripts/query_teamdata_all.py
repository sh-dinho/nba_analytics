# ============================================================
# File: scripts/query_teamdata_all.py
# Purpose: Query the unified teamdata_all table for multi-season feature engineering
# ============================================================

import sqlite3
import pandas as pd

# Path to your TeamData database
db_path = "../../Data/TeamData.sqlite"

# Connect to DB
con = sqlite3.connect(db_path)

# Load the master table
df = pd.read_sql_query("SELECT * FROM teamdata_all", con)

# Example 1: Filter by season
df_2022 = df[df["Season"] == "2022-23"]

# Example 2: Compute rolling averages across seasons
df_sorted = df.sort_values(["Season", "Date", "TEAM_ID"])
df_sorted["PTS_rolling_10"] = (
    df_sorted.groupby("TEAM_ID")["PTS"]
             .transform(lambda x: x.rolling(10, min_periods=1).mean())
)

# Example 3: Rest-day impact across seasons
df_sorted["RestImpact"] = df_sorted["Days-Rest-Home"] * df_sorted["PTS"]

# Example 4: Aggregate stats by season
season_summary = (
    df_sorted.groupby("Season")[["PTS", "AST", "REB"]]
             .mean()
             .reset_index()
)

print("Season summary (avg stats):")
print(season_summary)

# Close DB
con.close()