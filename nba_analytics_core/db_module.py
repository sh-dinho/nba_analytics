import json
import pandas as pd
import os
import sqlite3
import config

DB_PATH = config.DB_PATH

def connect():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)

def init_db():
    with connect() as con:
        cur = con.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS nba_games (
                game_id TEXT PRIMARY KEY,
                date TEXT,
                season INTEGER,
                home_team TEXT,
                away_team TEXT,
                home_score INTEGER,
                away_score INTEGER,
                winner TEXT
            )
        """)
        con.commit()
    # After ensuring table exists, refresh stats
    export_feature_stats()

def export_feature_stats(output_path="artifacts/feature_stats.json"):
    """Compute historical feature statistics and save to JSON."""
    with connect() as con:
        df = pd.read_sql("SELECT * FROM nba_games", con)

    if df.empty:
        print("No historical games found. Cannot export stats.")
        return

    numeric_df = df.select_dtypes(include=["number"])
    stats = {
        "mean": numeric_df.mean().to_dict(),
        "std": numeric_df.std().to_dict()
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=4)

    print(f"Feature stats exported to {output_path}")