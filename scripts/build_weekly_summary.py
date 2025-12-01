# File: scripts/build_weekly_summary.py
import os
import pandas as pd

os.makedirs("results", exist_ok=True)

def build_weekly_summary(notify=False):
    try:
        df = pd.read_csv("results/player_trends.csv")
    except FileNotFoundError:
        raise RuntimeError("Trend data not found. Run player_trends.py first.")

    # Check if *_change columns exist
    change_cols = ["PTS_change","REB_change","AST_change","TS_PCT_change"]
    raw_cols = ["points","rebounds","assists","TS_PCT"]

    if all(col in df.columns for col in change_cols):
        df["total_change"] = df[change_cols].sum(axis=1)
    else:
        df["total_change"] = df[raw_cols].sum(axis=1)

    summary = df.groupby("TEAM_ABBREVIATION").agg({
        "points":"mean",
        "rebounds":"mean",
        "assists":"mean",
        "TS_PCT":"mean",
        "total_change":"mean"
    }).reset_index()

    summary.to_csv("results/weekly_summary.csv", index=False)
    print("✅ Weekly summary saved to results/weekly_summary.csv")

if __name__ == "__main__":
    build_weekly_summary(notify=True)