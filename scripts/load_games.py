# scripts/load_games.py
import sqlite3
import pandas as pd
import yaml
import os

CONFIG = yaml.safe_load(open("config.yaml"))
DB_PATH = CONFIG["database"]["path"]

csv_path = "data/seed_nba_games.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSV not found at {csv_path}")

df = pd.read_csv(csv_path)
with sqlite3.connect(DB_PATH) as con:
    df.to_sql("nba_games", con, if_exists="append", index=False)
print("✔ Games loaded")