# scripts/load_games.py (Updated)
import sqlite3
import pandas as pd
import os
import logging
from config import DB_PATH # Use centralized config

csv_path = "data/seed_nba_games.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSV not found at {csv_path}")

try:
    df = pd.read_csv(csv_path)
    with sqlite3.connect(DB_PATH) as con:
        # NOTE: Using 'replace' here might be better if the seed file is authoritative
        df.to_sql("nba_games", con, if_exists="append", index=False) 
    logging.info(f"✔ {len(df)} games loaded from CSV into {DB_PATH}")
except Exception as e:
    logging.error(f"Failed to load seed games: {e}")