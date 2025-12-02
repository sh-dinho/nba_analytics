# File: scripts/generate_weekly_snapshots.py
import pandas as pd
import os
import argparse
import sys
import logging
import json
from datetime import datetime

REQUIRED_GAME_COLUMNS = {"date", "team", "points", "rebounds", "assists"}

# ----------------------------
# Logging setup
# ----------------------------
logger = logging.getLogger("weekly_snapshots")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
logger.addHandler(handler)


def _ensure_columns(df, required_cols, name):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def main(
    raw_file="data/games.csv",
    out_file="results/weekly_snapshots.csv",
    summary_file="results/weekly_snapshots_summary.csv",
    summary_top=5
):
    os.makedirs("results", exist_ok=True)

    if not os.path.exists(raw_file):
        raise FileNotFoundError(f"❌ Raw games file not found at {raw_file}. Place your source data there.")

    logger.info(f"📂 Loading raw game data from {raw_file}...")
    df = pd.read_csv(raw_file)

    # Validate required columns
    _ensure_columns(df, REQUIRED_GAME_COLUMNS, "games.csv")

    # Ensure date column is datetime
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        raise ValueError("❌ Invalid date values found in games.csv")

    # Add a 'week' column based on ISO calendar week
    df["week"] = df["date"].dt.isocalendar().week

    # Weekly snapshots: average stats per team per week
    snapshots = df.groupby(["team", "week"]).agg({
        "points": "mean",
        "rebounds": "mean",
        "assists": "mean"
    }).reset_index()

    # Add win/loss counts if available
    if {"home_points", "away_points"}.issubset(df.columns):
        if "opp_points" in df.columns:
            df["win"] = (df["points"] > df["opp_points"]).astype(int)
            win_summary = df.groupby(["team", "week"])["win"].sum().reset_index(name="wins")
            snapshots = snapshots.merge(win_summary, on=["team", "week"], how="left")

    # Save snapshots
    snapshots.to_csv(out_file, index=False)
    ts_out_file = out_file.replace(".csv", f"_{_timestamp()}.csv")
    snapshots.to_csv(ts_out_file, index=False)

    # Save summary (top teams by points per week)
    summary = snapshots.sort_values(["week", "points"], ascending=[True, False]).groupby("week").head(summary_top)
    summary.to_csv(summary_file, index=False)
    ts_summary_file = summary_file.replace(".csv", f"_{_timestamp()}.csv")
    summary.to_csv(ts_summary_file, index=False)

    logger.info(f"✅ Weekly snapshots saved to {out_file}")
    logger.info(f"📦 Timestamped backup saved to {ts_out_file}")
    logger.info(f"📊 Summary saved to {summary_file}")
    logger.info(f"📦 Timestamped backup saved to {ts_summary_file}")
    logger.info(f"🔎 Rows: {len(snapshots)}, Columns: {list(snapshots.columns)}")

    # Save metadata
    meta = {
        "generated_at": datetime.now().isoformat(),
        "rows": len(snapshots),
        "columns": snapshots.columns.tolist(),
        "raw_file": raw_file,
        "out_file": out_file,
        "summary_file": summary_file,
        "summary_top": summary_top
    }
    meta_file = "results/weekly_snapshots_meta.json"
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"🧾 Metadata saved to {meta_file}")

    return snapshots, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate weekly team snapshots from game data")
    parser.add_argument("--raw", type=str, default="data/games.csv", help="Path to raw games file")
    parser.add_argument("--export", type=str, default="results/weekly_snapshots.csv", help="Path to export snapshots file")
    parser.add_argument("--summary", type=str, default="results/weekly_snapshots_summary.csv", help="Path to export summary file")
    parser.add_argument("--summary-top", type=int, default=5, help="Number of top teams per week to include in summary")
    args = parser.parse_args()

    try:
        main(raw_file=args.raw, out_file=args.export, summary_file=args.summary, summary_top=args.summary_top)
    except Exception as e:
        logger.error(f"❌ Weekly snapshots generation failed: {e}")
        sys.exit(1)