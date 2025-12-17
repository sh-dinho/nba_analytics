# ============================================================
# File: src/ranking/bet.py
# Purpose: Betting recommendations based on predicted_win (v2.0)
# Version: 2.0
# Author: Your Team
# Date: December 2025
# ============================================================

import pandas as pd


def generate_betting_recommendations(df: pd.DataFrame, config_obj) -> pd.DataFrame:
    threshold = getattr(getattr(config_obj, "Betting", config_obj), "threshold", 0.6)
    cols = ["gameId", "homeTeam", "awayTeam", "predicted_win"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Betting recommendations missing columns: {missing}")
    recs = df.loc[df["predicted_win"] >= threshold, cols].copy()
    recs = recs.sort_values("predicted_win", ascending=False)
    return recs
