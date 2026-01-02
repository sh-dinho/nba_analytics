from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: ELO Rolling Features
# File: src/features/elo_rolling.py
# Author: Sadiq
#
# Description:
#     Adds rolling ELO features (last 5 and 10 games).
#     Assumes base ELO + opponent ELO have already been added
#     via add_elo_features().
# ============================================================

import pandas as pd


def add_elo_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds rolling ELO features.

    Required columns:
        - team
        - date
        - elo

    Produces:
        - elo_roll5
        - elo_roll10
    """
    out = df.copy()

    # Validate required columns
    required = {"team", "date", "elo"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"add_elo_rolling_features missing columns: {missing}")

    # Ensure proper ordering
    out = out.sort_values(["team", "date"]).reset_index(drop=True)

    grouped = out.groupby("team")

    # Leakage-safe rolling ELO (shift → rolling → mean)
    out["elo_roll5"] = (
        grouped["elo"]
        .shift(1)
        .rolling(window=5, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
        .astype("float32")
    )

    out["elo_roll10"] = (
        grouped["elo"]
        .shift(1)
        .rolling(window=10, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
        .astype("float32")
    )

    return out
