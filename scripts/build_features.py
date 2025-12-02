# File: scripts/build_features.py
import os
import pandas as pd
import numpy as np
from datetime import datetime
import json
import logging

logger = logging.getLogger("build_features")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
logger.addHandler(handler)

FEATURES_DIR = "results"
PLAYER_STATS_FILE = "data/player_stats.csv"

REQUIRED_COLS = {"playerid", "pts", "reb", "ast", "ts_pct", "date", "team_id"}

def _ensure_columns(df, required_cols, name):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")

def _scale_numeric(df, exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = []
    numeric_cols = df.select_dtypes(include=["number"]).columns.difference(exclude_cols)
    for col in numeric_cols:
        mean, std = df[col].mean(), df[col].std(ddof=0)
        df[f"{col}_z"] = (df[col] - mean) / std if std > 0 else 0
    return df

def build_features(scale=True):
    if not os.path.exists(PLAYER_STATS_FILE):
        raise RuntimeError(f"{PLAYER_STATS_FILE} not found. Run fetch_player_stats_parallel.py first.")

    df = pd.read_csv(PLAYER_STATS_FILE)
    df.columns = [c.lower() for c in df.columns]  # normalize
    _ensure_columns(df, REQUIRED_COLS, PLAYER_STATS_FILE)

    # Convert date
    df["date"] = pd.to_datetime(df["date"])

    # Compute simple per-game features
    df["total_stats"] = df[["pts", "reb", "ast"]].sum(axis=1)
    df["efficiency"] = df["pts"] + 0.4*df["reb"] + 0.7*df["ast"]

    # Optional z-score scaling
    if scale:
        df = _scale_numeric(df, exclude_cols=["playerid", "team_id", "date"])

    os.makedirs(FEATURES_DIR, exist_ok=True)
    out_file = os.path.join(FEATURES_DIR, "player_features.csv")
    df.to_csv(out_file, index=False)
    logger.info(f"✅ Player features saved to {out_file}")

    # Metadata
    meta = {
        "generated_at": datetime.now().isoformat(),
        "rows": len(df),
        "columns": df.columns.tolist(),
        "scaled": scale
    }
    meta_file = os.path.join(FEATURES_DIR, "player_features_meta.json")
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"🧾 Metadata saved to {meta_file}")

    return df
