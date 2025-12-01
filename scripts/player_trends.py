# File: scripts/player_trends.py
import pandas as pd
import os

def main():
    os.makedirs("results", exist_ok=True)

    # Example: load raw player stats (adjust path to your actual source)
    raw_file = "data/player_stats.csv"
    if not os.path.exists(raw_file):
        raise FileNotFoundError("Raw player stats file not found. Place your source data in data/player_stats.csv")

    df = pd.read_csv(raw_file)

    # Convert date to datetime and add week number
    df["date"] = pd.to_datetime(df["date"])
    df["week"] = df["date"].dt.isocalendar().week

    # Example trend calculation: rolling averages of points, rebounds, assists
    df = df.sort_values(["player", "date"])
    df["points_trend"] = df.groupby("player")["points"].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df["rebounds_trend"] = df.groupby("player")["rebounds"].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df["assists_trend"] = df.groupby("player")["assists"].transform(lambda x: x.rolling(3, min_periods=1).mean())

    # Aggregate by team and week to get team-level trends
    trends = df.groupby(["team", "week"]).agg({
        "points_trend": "mean",
        "rebounds_trend": "mean",
        "assists_trend": "mean"
    }).reset_index()

    out_file = "results/player_trends.csv"
    trends.to_csv(out_file, index=False)
    print(f"✅ Player trends saved to {out_file}")

if __name__ == "__main__":
    main()