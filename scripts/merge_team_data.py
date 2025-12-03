# ============================================================
# File: scripts/merge_team_data.py
# Purpose: Merge all season-level team tables into one master table (teamdata_all)
# ============================================================

import os
import sqlite3
import pandas as pd

# Path to your TeamData database
db_path = "../../Data/TeamData.sqlite"

# Connect to DB
con = sqlite3.connect(db_path)

# Get list of all season tables
cursor = con.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'teamdata_%'")
season_tables = [row[0] for row in cursor.fetchall()]

print(f"Found {len(season_tables)} season tables: {season_tables}")

# Merge all season tables into one DataFrame
frames = []
for table in season_tables:
    df = pd.read_sql_query(f"SELECT * FROM {table}", con)
    df["Season"] = table.replace("teamdata_", "")
    frames.append(df)

master_df = pd.concat(frames, ignore_index=True)

# Save to master table
master_df.to_sql("teamdata_all", con, if_exists="replace", index=False)

print(f"✅ Merged {len(season_tables)} tables into teamdata_all with {len(master_df)} rows")

con.close()