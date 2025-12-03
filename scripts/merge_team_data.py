# ============================================================
# File: scripts/merge_team_data.py
# Purpose: Merge team tables into one master table (teamdata_all)
# ============================================================

import os
import sqlite3
import pandas as pd
import datetime
import argparse

# Resolve DB path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
relative_path = os.path.join(BASE_DIR, "Data", "TeamData.sqlite")

# Fallback absolute path (edit if needed)
fallback_path = r"C:\Users\Mohamadou\projects\nba_analytics\Data\TeamData.sqlite"

if os.path.exists(relative_path):
    db_path = relative_path
else:
    db_path = fallback_path

print("Resolved DB path:", db_path)
print("Exists?", os.path.exists(db_path))

def get_current_season_label() -> str:
    """Determine current NBA season label based on today's date."""
    today = datetime.date.today()
    year = today.year
    if today.month >= 10:  # Oct–Dec → season spans current year → next year
        return f"{year}_{year+1}"
    else:  # Jan–Jun → season spans previous year → current year
        return f"{year-1}_{year}"

def merge_current_season(con):
    """Merge only the current season table into teamdata_all."""
    season_label = get_current_season_label()
    season_table = f"teamdata_{season_label}"

    cursor = con.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (season_table,))
    row = cursor.fetchone()

    if not row:
        print(f"⚠️ No table found for current season: {season_table}")
        return None

    print(f"Found current season table: {season_table}")
    df = pd.read_sql_query(f"SELECT * FROM {season_table}", con)
    df["Season"] = season_label
    return df

def merge_all_seasons(con):
    """Merge all season tables into teamdata_all."""
    cursor = con.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'teamdata_%'")
    season_tables = [row[0] for row in cursor.fetchall()]

    if not season_tables:
        print("⚠️ No season tables found in the database.")
        return None

    print(f"Found {len(season_tables)} season tables: {season_tables}")
    frames = []
    for table in season_tables:
        df = pd.read_sql_query(f"SELECT * FROM {table}", con)
        df["Season"] = table.replace("teamdata_", "")
        frames.append(df)

    return pd.concat(frames, ignore_index=True)

def main(use_all: bool):
    con = sqlite3.connect(db_path)

    if use_all:
        master_df = merge_all_seasons(con)
    else:
        master_df = merge_current_season(con)

    if master_df is None:
        con.close()
        return

    # Save to master table
    master_df.to_sql("teamdata_all", con, if_exists="replace", index=False)

    # Optional: add index for faster queries
    con.execute("CREATE INDEX IF NOT EXISTS idx_teamdata_all_season ON teamdata_all(Season)")

    print(f"✅ Merged into teamdata_all with {len(master_df)} rows")

    con.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge team tables into teamdata_all")
    parser.add_argument("--all", action="store_true", help="Merge all seasons instead of just current season")
    args = parser.parse_args()
    main(use_all=args.all)