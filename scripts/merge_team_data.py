# ============================================================
# File: scripts/merge_team_data.py
# Purpose: Merge all season-level team tables into one master table (teamdata_all)
# ============================================================

import sqlite3
import pandas as pd

# Path to your TeamData database
db_path = "../../Data/TeamData.sqlite"

# Connect to DB
con = sqlite3.connect(db_path)
cursor = con.cursor()

# Get list of all season tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'teamdata_%'")
season_tables = [row[0] for row in cursor.fetchall()]

print(f"Found {len(season_tables)} season tables: {season_tables}")

frames = []
for table in season_tables:
    df = pd.read_sql_query(f"SELECT * FROM {table}", con)

    # Normalize season name (strip prefixes/suffixes)
    season_name = table.replace("teamdata_", "").replace("_new", "")
    df["Season"] = season_name

    frames.append(df)
    print(f"{table}: {len(df)} rows")

# Ensure consistent schema across all frames
base_cols = frames[0].columns
frames = [f.reindex(columns=base_cols, fill_value=None) for f in frames]

# Merge into one master DataFrame
master_df = pd.concat(frames, ignore_index=True)

# Save to master table
master_df.to_sql("teamdata_all", con, if_exists="replace", index=False)

# Add index for performance
cursor.execute("CREATE INDEX IF NOT EXISTS idx_team_date ON teamdata_all (Date, TEAM_ID)")

print(f"✅ Merged {len(season_tables)} tables into teamdata_all with {len(master_df)} rows")

con.close()