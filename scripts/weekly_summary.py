# File: scripts/weekly_summary.py
import pandas as pd
import os

def main():
    os.makedirs("results", exist_ok=True)

    # Example: load player snapshots and build weekly summary
    snapshots_file = "results/weekly_snapshots.csv"
    if not os.path.exists(snapshots_file):
        raise FileNotFoundError("Run generate_weekly_snapshots.py first.")

    df = pd.read_csv(snapshots_file)

    # Example aggregation: average stats per team per week
    summary = df.groupby(["team", "week"]).mean().reset_index()

    out_file = "results/weekly_summary.csv"
    summary.to_csv(out_file, index=False)
    print(f"✅ Weekly summary saved to {out_file}")

if __name__ == "__main__":
    main()