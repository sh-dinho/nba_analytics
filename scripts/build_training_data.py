# File: scripts/build_training_data.py
import os
import sys
import json
import logging
from datetime import datetime

import pandas as pd
import numpy as np

# ----------------------------
# Logging setup
# ----------------------------
logger = logging.getLogger("build_training_data")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
logger.addHandler(handler)

# ----------------------------
# Constants
# ----------------------------
REQUIRED_FEATURE_COLUMNS = {"game_id", "date", "home_team", "away_team", "home_points", "away_points"}
EXCLUDE_FROM_SCALING = {"home_win", "target_margin"}

# ----------------------------
# Helper functions
# ----------------------------
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

def _compute_streaks(df: pd.DataFrame) -> pd.DataFrame:
    """Compute winning streaks per team across games."""
    streaks = []

    teams = pd.concat([df["home_team"], df["away_team"]]).unique()
    for team in teams:
        team_games = df[(df["home_team"] == team) | (df["away_team"] == team)].sort_values("date")
        wins = ((team_games["home_team"] == team) & (team_games["home_points"] > team_games["away_points"])) | \
               ((team_games["away_team"] == team) & (team_games["away_points"] > team_games["home_points"]))
        streak = 0
        streak_list = []
        for w in wins:
            streak = streak + 1 if w else 0
            streak_list.append(streak)
        team_games = team_games.copy()
        team_games["streak"] = streak_list
        streaks.append(team_games)

    return pd.concat(streaks, ignore_index=True)

# ----------------------------
# Main function
# ----------------------------
def build_training_data(features_file="features/training_features.csv",
                        out_file="features/training_data.csv",
                        scale=True):
    os.makedirs("features", exist_ok=True)

    if not os.path.exists(features_file):
        raise FileNotFoundError(f"{features_file} not found. Run build_features.py first.")

    logger.info("📂 Loading features...")
    df = pd.read_csv(features_file)

    missing_cols = REQUIRED_FEATURE_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"Features file is missing required columns: {missing_cols}")

    # Parse date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # Target variables
    df["home_win"] = (df["home_points"] > df["away_points"]).astype(int)
    df["target_margin"] = df["home_points"] - df["away_points"]

    # Fill missing numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.difference(EXCLUDE_FROM_SCALING)
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # Compute streaks
    logger.info("📈 Computing winning streak features...")
    df_streaks = _compute_streaks(df)
    df = df.merge(df_streaks[["game_id", "streak"]], on="game_id", how="left")

    # Optional scaling
    if scale:
        logger.info("📊 Scaling numeric features...")
        df = _scale_numeric(df, exclude_cols=EXCLUDE_FROM_SCALING)

    # Save datasets
    df.to_csv(out_file, index=False)
    ts_file = f"features/training_data_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    df.to_csv(ts_file, index=False)

    logger.info(f"✅ Training dataset saved to {out_file}")
    logger.info(f"📦 Timestamped backup saved to {ts_file}")
    logger.info(f"Rows: {len(df)}, Columns: {len(df.columns)}")

    # Metadata
    meta = {
        "generated_at": datetime.now().isoformat(),
        "rows": len(df),
        "columns": df.columns.tolist(),
        "features_file": features_file,
        "scaled": scale,
        "streak_feature": "streak"
    }
    meta_file = "features/training_data_meta.json"
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"🧾 Metadata saved to {meta_file}")

    return df

# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build model-ready training dataset")
    parser.add_argument("--features", type=str, default="features/training_features.csv")
    parser.add_argument("--out", type=str, default="features/training_data.csv")
    parser.add_argument("--no-scale", action="store_true", help="Disable z-score scaling")

    args = parser.parse_args()

    try:
        build_training_data(features_file=args.features, out_file=args.out, scale=not args.no_scale)
    except Exception as e:
        logger.error(f"❌ Training data build failed: {e}")
        sys.exit(1)
