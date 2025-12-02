# File: scripts/build_weekly_summary.py
import os
import pandas as pd

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
    return df

def build_weekly_summary(notify=False):
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

    # Check if *_change columns exist
    change_cols = ["PTS_change", "REB_change", "AST_change", "TS_PCT_change"]
    raw_cols = ["points", "rebounds", "assists", "TS_PCT"]

    if all(col in df.columns for col in change_cols):
        df["total_change"] = df[change_cols].sum(axis=1)
    else:
        df["total_change"] = df[raw_cols].sum(axis=1)

    # Add week number for grouping
    df["week"] = df["date"].dt.isocalendar().week

    # Weekly summary aggregation
    summary = df.groupby(["TEAM_ABBREVIATION", "week"]).agg({
        "points": "mean",
        "rebounds": "mean",
        "assists": "mean",
        "TS_PCT": "mean",
        "total_change": "mean"
    }).reset_index()

    # Rolling weekly comparison (difference vs previous week)
    summary = summary.sort_values(["TEAM_ABBREVIATION", "week"])
    for col in ["points", "rebounds", "assists", "TS_PCT", "total_change"]:
        summary[f"{col}_diff_prev_week"] = summary.groupby("TEAM_ABBREVIATION")[col].diff()

    # Feature scaling
    summary = _scale_numeric(summary, exclude_cols=["TEAM_ABBREVIATION", "week"])

    # Save summary
    out_file = "results/weekly_summary.csv"
    summary.to_csv(out_file, index=False)

    if notify:
        print(f"✅ Weekly summary built and saved to {out_file}")
        print(f"📊 Rows: {len(summary)}, Columns: {len(summary.columns)}")
        print(f"🔎 Columns: {list(summary.columns)}")

if __name__ == "__main__":
    build_weekly_summary(notify=True)