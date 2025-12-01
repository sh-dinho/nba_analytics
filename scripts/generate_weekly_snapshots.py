# File: scripts/generate_weekly_snapshots.py
import pandas as pd
import os
from datetime import datetime

def main():
    os.makedirs("results", exist_ok=True)

    # Example: load raw player/game data
    raw_file = "data/games.csv"   # adjust to your actual raw data source
    if not os.path.exists(raw_file):
        raise FileNotFoundError("Raw games file not found. Place your source data in data/games.csv")

    df = pd.read_csv(raw_file)

    # Add a 'week' column based on date
    df["date"] = pd.to_datetime(df["date"])
    df["week"] = df["date"].dt.isocalendar().week

    # Example snapshot: average stats per team per week
    snapshots = df.groupby(["team", "week"]).mean().reset_index()

    out_file = "results/weekly_snapshots.csv"
    snapshots.to_csv(out_file, index=False)
    print(f"✅ Weekly snapshots saved to {out_file}")

if __name__ == "__main__":
    main()