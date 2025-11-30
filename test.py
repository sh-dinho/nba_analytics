# verify_nba_games.py
import sqlite3
import yaml
import pandas as pd

# Load DB path from config
with open("config.yaml") as f:
    CONFIG = yaml.safe_load(f)
DB_PATH = CONFIG["database"]["path"]

def verify_nba_games():
    with sqlite3.connect(DB_PATH) as con:
        # Show schema
        print("=== Schema ===")
        schema = con.execute("PRAGMA table_info(nba_games);").fetchall()
        for col in schema:
            print(col)

        # Show row count
        count = con.execute("SELECT COUNT(*) FROM nba_games;").fetchone()[0]
        print(f"\nTotal rows: {count}")

        # Show first 5 rows
        df = pd.read_sql("SELECT * FROM nba_games LIMIT 5;", con)
        print("\n=== Sample rows ===")
        print(df)

if __name__ == "__main__":
    verify_nba_games()