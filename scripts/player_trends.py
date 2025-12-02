# File: scripts/player_trends.py
import pandas as pd
import os
import argparse
import logging
import json
from datetime import datetime

REQUIRED_PLAYER_COLUMNS = {"date", "player", "team", "points", "rebounds", "assists"}

# ----------------------------
# Logging setup
# ----------------------------
logger = logging.getLogger("player_trends")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
logger.addHandler(handler)


def _ensure_columns(df, required_cols, name):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def main(raw_file="data/player_stats.csv", out_file="results/player_trends.csv", window=3):
    os.makedirs("results", exist_ok=True)

    if not os.path.exists(raw_file):
        raise FileNotFoundError(f"❌ Raw player stats file not found at {raw_file}.")

    logger.info(f"📂 Loading raw player stats from {raw_file}...")
    df = pd.read_csv(raw_file)
    _ensure_columns(df, REQUIRED_PLAYER_COLUMNS, "player_stats.csv")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        raise ValueError("❌ Invalid date values found in player_stats.csv")
    df["week"] = df["date"].dt.isocalendar().week

    df = df.sort_values(["player", "date"])

    # Rolling averages for trends
    df["points_trend"] = df.groupby("player")["points"].transform(lambda x: x.rolling(window, min_periods=1).mean())
    df["rebounds_trend"] = df.groupby("player")["rebounds"].transform(lambda x: x.rolling(window, min_periods=1).mean())
    df["assists_trend"] = df.groupby("player")["assists"].transform(lambda x: x.rolling(window, min_periods=1).mean())

    # Compute week-over-week differences
    for col in ["points_trend", "rebounds_trend", "assists_trend"]:
        df[f"{col}_diff"] = df.groupby("player")[col].diff()

    # Aggregate by team and week to get team-level trends
    trends = df.groupby(["team", "week"]).agg({
        "points_trend": "mean",
        "rebounds_trend": "mean",
        "assists_trend": "mean",
        "points_trend_diff": "mean",
        "rebounds_trend_diff": "mean",
        "assists_trend_diff": "mean"
    }).reset_index()

    # Z-score normalization per week
    numeric_cols = ["points_trend", "rebounds_trend", "assists_trend",
                    "points_trend_diff", "rebounds_trend_diff", "assists_trend_diff"]
    for col in numeric_cols:
        trends[f"{col}_z"] = trends.groupby("week")[col].transform(
            lambda x: (x - x.mean()) / (x.std(ddof=0) or 1)
        )

    # Save detailed trends
    trends.to_csv(out_file, index=False)
    ts_file = out_file.replace(".csv", f"_{_timestamp()}.csv")
    trends.to_csv(ts_file, index=False)

    # Save summary (top teams by points trend per week)
    summary_file = out_file.replace(".csv", "_summary.csv")
    summary = trends.sort_values(["week", "points_trend"], ascending=[True, False]).groupby("week").head(5)
    summary.to_csv(summary_file, index=False)
    ts_summary_file = summary_file.replace(".csv", f"_{_timestamp()}.csv")
    summary.to_csv(ts_summary_file, index=False)

    logger.info(f"✅ Player trends saved to {out_file}")
    logger.info(f"📦 Timestamped backup saved to {ts_file}")
    logger.info(f"📊 Summary saved to {summary_file}")
    logger.info(f"📦 Timestamped summary backup saved to {ts_summary_file}")
    logger.info(f"🔎 Rows: {len(trends)}, Columns: {list(trends.columns)}")

    # Save metadata
    meta = {
        "generated_at": datetime.now().isoformat(),
        "rows": len(trends),
        "columns": trends.columns.tolist(),
        "raw_file": raw_file,
        "out_file": out_file,
        "summary_file": summary_file,
        "window": window
    }
    meta_file = out_file.replace(".csv", "_meta.json")
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"🧾 Metadata saved to {meta_file}")

    return trends, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate player and team trends from player stats")
    parser.add_argument("--raw", type=str, default="data/player_stats.csv", help="Path to raw player stats file")
    parser.add_argument("--export", type=str, default="results/player_trends.csv", help="Path to export trends file")
    parser.add_argument("--window", type=int, default=3, help="Rolling window size for trend calculation")
    args = parser.parse_args()

    try:
        main(raw_file=args.raw, out_file=args.export, window=args.window)
    except Exception as e:
        logger.error(f"❌ Player trends generation failed: {e}")
        exit(1)
