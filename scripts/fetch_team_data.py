# ============================================================
# File: scripts/merge_team_data.py
# Purpose: Merge all season-level team tables into one master table
#          with logging, archiving, robust error handling, and skip-empty logic
# ============================================================

import os
import sqlite3
import pandas as pd
import datetime
from pathlib import Path

# --- Configuration ---
DB_PATH = "../../Data/TeamData.sqlite"
ARCHIVE_DIR = "../../Data/archive"
MASTER_TABLE = "teamdata_all"
ARCHIVE_PREFIX = "teamdata_backup"

Path(ARCHIVE_DIR).mkdir(parents=True, exist_ok=True)

# --- Connect to SQLite DB ---
try:
    con = sqlite3.connect(DB_PATH)
    cursor = con.cursor()
except sqlite3.Error as e:
    raise RuntimeError(f"❌ Failed to connect to database {DB_PATH}: {e}")

# --- List all season tables ---
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'teamdata_%'")
season_tables = [row[0] for row in cursor.fetchall()]

if not season_tables:
    raise RuntimeError("❌ No season tables found in the database.")

print(f"Found {len(season_tables)} season tables: {season_tables}")

# --- Archive existing master table if it exists ---
cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{MASTER_TABLE}'")
if cursor.fetchone():
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_name = f"{ARCHIVE_PREFIX}_{ts}"
    cursor.execute(f"ALTER TABLE {MASTER_TABLE} RENAME TO {archive_name}")
    con.commit()
    print(f"📦 Archived existing master table '{MASTER_TABLE}' → '{archive_name}'")

# --- Merge all season tables ---
frames = []
row_counts = {}
skipped_tables = []

for table in season_tables:
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table}", con)
        if df.empty:
            print(f"⚠️ Skipping empty table {table}")
            skipped_tables.append(table)
            continue
        df["Season"] = table.replace("teamdata_", "")
        frames.append(df)
        row_counts[table] = len(df)
        print(f"✅ Loaded {table} with {len(df)} rows")
    except Exception as e:
        print(f"⚠️ Failed to read table {table}: {e}")
        skipped_tables.append(table)

if not frames:
    raise RuntimeError("❌ No non-empty tables loaded. Aborting merge.")

master_df = pd.concat(frames, ignore_index=True)

# --- Save to master table ---
try:
    master_df.to_sql(MASTER_TABLE, con, if_exists="replace", index=False)
    print(f"✅ Merged {len(frames)} tables into '{MASTER_TABLE}' ({len(master_df)} rows)")
    if skipped_tables:
        print(f"⚠️ Skipped {len(skipped_tables)} tables: {skipped_tables}")
except Exception as e:
    raise RuntimeError(f"❌ Failed to write master table {MASTER_TABLE}: {e}")

# --- Optional: Save summary CSV ---
summary_file = Path(DB_PATH).parent / "merge_teamdata_summary.csv"
summary_df = pd.DataFrame([{"table": k, "rows": v} for k, v in row_counts.items()])
summary_df["total_rows"] = len(master_df)
summary_df["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
summary_df["skipped_tables"] = ",".join(skipped_tables)
summary_df.to_csv(summary_file, index=False)
print(f"📊 Merge summary saved to {summary_file}")

# --- Close connection ---
con.close()
