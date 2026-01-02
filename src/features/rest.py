from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Rest Features
# File: src/features/rest.py
# Author: Sadiq
#
# Description:
#     Rest-related features for team-game data:
#     - rest_days
#     - is_b2b (0 or 1)
# ============================================================

import pandas as pd


def add_rest_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Validate required columns
    required = {"team", "date"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"add_rest_features missing columns: {missing}")

    # Ensure datetime
    out["date"] = pd.to_datetime(out["date"])

    # Sort for rolling operations
    out = out.sort_values(["team", "date"]).reset_index(drop=True)

    # Previous game date
    out["prev_date"] = out.groupby("team")["date"].shift(1)

    # Days since previous game
    # First game of season gets rest_days = 10 (safe default)
    out["rest_days"] = (
        (out["date"] - out["prev_date"]).dt.days
        .fillna(10)
        .astype(int)
    )

    # Back-to-back indicator
    # True B2B = rest_days == 1
    out["is_b2b"] = (out["rest_days"] == 1).astype(int)

    return out.drop(columns=["prev_date"])
