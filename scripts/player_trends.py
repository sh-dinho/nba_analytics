# File: scripts/player_trends.py
import pandas as pd
import os
import argparse

REQUIRED_PLAYER_COLUMNS = {"date", "player", "team", "points", "rebounds", "assists"}

def _ensure_columns(df, required_cols, name):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")

def main(raw_file="data/player_stats.csv", out_file="results/player_trends.csv", window=3):
    os.makedirs("results", exist_ok=True)

    if not os.path.exists(raw_file):
        raise FileNotFoundError(f"❌ Raw player stats file not found at {raw_file}. Place your source data there.")

    print(f"📂 Loading raw player stats from {raw_file}...")
    df = pd.read_csv(raw_file)

    # Validate required columns
    _ensure_columns(df, REQUIRED_PLAYER_COLUMNS, "player_stats.csv")

    # Convert date to datetime and add week number
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        raise ValueError("❌ Invalid date values found in player_stats.csv")
    df["week"] = df["date"].dt.isocalendar().week

    # Sort for rolling calculations
    df = df.sort_values(["player", "date"])

    # Rolling averages for trends
    df["points_trend"] = df.groupby("player")["points"].transform(lambda x: x.rolling(window, min_periods=1).mean())
    df["rebounds_trend"] = df.groupby("player")["rebounds"].transform(lambda x: x.rolling(window, min_periods=1).mean())
    df["assists_trend"] = df.groupby("player")["assists"].transform(lambda x: x.rolling(window, min_periods=1).mean())

    # Aggregate by team and week to get team-level trends
    trends = df.groupby(["team", "week"]).agg({
        "points_trend": "mean",
        "rebounds_trend": "mean",
        "assists_trend": "mean"
    }).reset_index()

    # Save detailed trends
    trends.to_csv(out_file, index=False)

    # Save summary (top teams by points trend per week)
    summary_file = out_file.replace(".csv", "_summary.csv")
    summary = trends.sort_values(["week", "points_trend"], ascending=[True, False]).groupby("week").head(5)
    summary.to_csv(summary_file, index=False)

    print(f"✅ Player trends saved to {out_file}")
    print(f"📊 Summary saved to {summary_file}")
    print(f"🔎 Rows: {len(trends)}, Columns: {list(trends.columns)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate player and team trends from player stats")
    parser.add_argument("--raw", type=str, default="data/player_stats.csv", help="Path to raw player stats file")
    parser.add_argument("--export", type=str, default="results/player_trends.csv", help="Path to export trends file")
    parser.add_argument("--window", type=int, default=3, help="Rolling window size for trend calculation")
    args = parser.parse_args()

    main(raw_file=args.raw, out_file=args.export, window=args.window)