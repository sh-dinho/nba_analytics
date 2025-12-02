# File: scripts/build_training_data.py
import pandas as pd
import os
import sys
import logging
from datetime import datetime
import json

REQUIRED_WEEKLY_COLUMNS = {"team", "date", "result"}
REQUIRED_TRENDS_COLUMNS = {"team", "date"}

# ----------------------------
# Logging setup
# ----------------------------
logger = logging.getLogger("build_training_data")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
logger.addHandler(handler)


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
            logger.warning(f"Skipping scaling for {col} (std=0)")
    return df


def build_training_data(
    weekly_file="results/weekly_summary.csv",
    trends_file="results/player_trends.csv",
    out_file="features/training_data.csv",
    scale=True
):
    os.makedirs("features", exist_ok=True)

    if not os.path.exists(weekly_file) or not os.path.exists(trends_file):
        raise FileNotFoundError("Weekly summary or player trends file not found. Run those scripts first.")

    logger.info("📂 Loading input files...")
    weekly = pd.read_csv(weekly_file)
    trends = pd.read_csv(trends_file)

    # Validate columns
    _ensure_columns(weekly, REQUIRED_WEEKLY_COLUMNS, "weekly_summary.csv")
    _ensure_columns(trends, REQUIRED_TRENDS_COLUMNS, "player_trends.csv")

    # Convert date to datetime
    weekly["date"] = pd.to_datetime(weekly["date"], errors="coerce")
    trends["date"] = pd.to_datetime(trends["date"], errors="coerce")

    logger.info("🔗 Merging weekly summary with player trends...")
    df = weekly.merge(trends, on=["team", "date"], how="left")

    # Add target column (win/loss)
    df["target_win"] = (weekly["result"] == "W").astype(int)

    # Optional: margin of victory target
    if {"points_for", "points_against"}.issubset(weekly.columns):
        df["target_margin"] = weekly["points_for"] - weekly["points_against"]

    # Feature scaling (z-score normalization)
    if scale:
        logger.info("📊 Scaling numeric features...")
        df = _scale_numeric(df, exclude_cols=["target_win", "target_margin"])

    # Save training dataset
    df.to_csv(out_file, index=False)
    ts_out = f"features/training_data_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    df.to_csv(ts_out, index=False)

    logger.info(f"✅ Training dataset built and saved to {out_file}")
    logger.info(f"📦 Timestamped backup saved to {ts_out}")
    logger.info(f"Rows: {len(df)}, Columns: {len(df.columns)}")

    # Save metadata
    meta = {
        "generated_at": datetime.now().isoformat(),
        "rows": len(df),
        "columns": df.columns.tolist(),
        "weekly_file": weekly_file,
        "trends_file": trends_file,
        "scaled": scale
    }
    meta_file = "features/training_data_meta.json"
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"🧾 Metadata saved to {meta_file}")

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build training dataset from weekly summary and player trends")
    parser.add_argument("--weekly", type=str, default="results/weekly_summary.csv")
    parser.add_argument("--trends", type=str, default="results/player_trends.csv")
    parser.add_argument("--out", type=str, default="features/training_data.csv")
    parser.add_argument("--no-scale", action="store_true", help="Disable z-score scaling")

    args = parser.parse_args()

    try:
        build_training_data(
            weekly_file=args.weekly,
            trends_file=args.trends,
            out_file=args.out,
            scale=not args.no_scale
        )
    except Exception as e:
        logger.error(f"❌ Training data build failed: {e}")
        sys.exit(1)