# File: scripts/build_weekly_summary.py
import os
import pandas as pd
import json
from datetime import datetime
import numpy as np

# Ensure results directory exists
os.makedirs("results", exist_ok=True)

REQUIRED_TRENDS_COLUMNS = {"TEAM_ABBREVIATION", "points", "rebounds", "assists", "TS_PCT"}

def _ensure_columns(df, required_cols, name):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")

def _scale_numeric(df, exclude_cols=None):
    """Standardize numeric columns (z-score)."""
    if exclude_cols is None:
        exclude_cols = []
    numeric_cols = df.select_dtypes(include=["number"]).columns.difference(exclude_cols)
    for col in numeric_cols:
        mean, std = df[col].mean(), df[col].std(ddof=0)
        if std > 0:
            df[f"{col}_z"] = (df[col] - mean) / std
        else:
            df[f"{col}_z"] = 0
    return df

def build_weekly_summary(notify=False, scale=True):
    # Load player trends
    try:
        df = pd.read_csv("results/player_trends.csv")
    except FileNotFoundError:
        raise RuntimeError("Trend data not found. Run player_trends.py first.")

    # Validate required columns
    _ensure_columns(df, REQUIRED_TRENDS_COLUMNS, "player_trends.csv")

    # Ensure date column exists and is datetime
    if "date" not in df.columns:
        raise ValueError("player_trends.csv must contain a 'date' column")
    df["date"] = pd.to_datetime(df["date"])

    # Handle change columns if present
    change_cols = ["PTS_change", "REB_change", "AST_change", "TS_PCT_change"]
    raw_cols = ["points", "rebounds", "assists", "TS_PCT"]

    df["total_change"] = df[change_cols].sum(axis=1) if all(c in df.columns for c in change_cols) else df[raw_cols].sum(axis=1)

    # Add week number
    df["week"] = df["date"].dt.isocalendar().week
    df["year"] = df["date"].dt.isocalendar().year

    # Weekly aggregation
    summary = df.groupby(["TEAM_ABBREVIATION", "year", "week"]).agg({
        "points": "mean",
        "rebounds": "mean",
        "assists": "mean",
        "TS_PCT": "mean",
        "total_change": "mean"
    }).reset_index()

    # Compute weekly differences vs previous week
    summary = summary.sort_values(["TEAM_ABBREVIATION", "year", "week"])
    for col in ["points", "rebounds", "assists", "TS_PCT", "total_change"]:
        summary[f"{col}_diff_prev_week"] = summary.groupby("TEAM_ABBREVIATION")[col].diff()

    # Compute rolling 3-week trend features
    for col in ["points", "rebounds", "assists", "TS_PCT", "total_change"]:
        summary[f"{col}_roll3_mean"] = summary.groupby("TEAM_ABBREVIATION")[col].transform(lambda x: x.rolling(3, min_periods=1).mean())
        summary[f"{col}_roll3_std"] = summary.groupby("TEAM_ABBREVIATION")[col].transform(lambda x: x.rolling(3, min_periods=1).std().fillna(0))

    # Optional scaling
    if scale:
        summary = _scale_numeric(summary, exclude_cols=["TEAM_ABBREVIATION", "week", "year"])

    # Save weekly summary
    out_file = "results/weekly_summary.csv"
    summary.to_csv(out_file, index=False)

    # Save metadata
    meta = {
        "generated_at": datetime.now().isoformat(),
        "rows": len(summary),
        "columns": summary.columns.tolist(),
        "source_file": "results/player_trends.csv",
        "scaled": scale
    }
    meta_file = "results/weekly_summary_meta.json"
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)

    if notify:
        print(f"✅ Weekly summary built and saved to {out_file}")
        print(f"📦 Metadata saved to {meta_file}")
        print(f"📊 Rows: {len(summary)}, Columns: {len(summary.columns)}")

    return summary

# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build weekly summary from player trends")
    parser.add_argument("--no-scale", action="store_true", help="Disable z-score scaling")
    args = parser.parse_args()

    build_weekly_summary(notify=True, scale=not args.no_scale)
