from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Rest Features
# File: src/features/rest.py
# Author: Sadiq
#
# Description:
#     Rest-related features for team-game data:
#     previous game date, rest_days, and back-to-back flag.
# ============================================================

import pandas as pd


def add_rest_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds rest-related features:
        - rest_days (days since previous game)
        - is_b2b (back-to-back indicator)

    Required columns:
        - team
        - date
    """
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values(["team", "date"])

    # Previous game date per team
    out["prev_date"] = out.groupby("team")["date"].shift()

    # Days of rest (default 10 for first game)
    out["rest_days"] = (
        (out["date"] - out["prev_date"])
        .dt.days
        .fillna(10)
        .astype("int16")
    )

    # Back-to-back indicator
    out["is_b2b"] = (out["rest_days"] == 1).astype("int8")

    return out.drop(columns=["prev_date"])