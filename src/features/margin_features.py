from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Margin Features
# File: src/features/margin_features.py
# Author: Sadiq
#
# Description:
#     Adds scoring margin features for team-game rows:
#     - margin
#     - margin_rolling_5
#     - margin_rolling_10
# ============================================================

import pandas as pd


def add_margin_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds scoring margin features.

    Required columns:
        - team
        - date
        - score
        - opp_score
    """
    out = df.copy()

    # Validate required columns
    required = {"team", "date", "score", "opp_score"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"add_margin_features missing columns: {missing}")

    # Ensure datetime
    out["date"] = pd.to_datetime(out["date"])

    # Sort for rolling operations
    out = out.sort_values(["team", "date"]).reset_index(drop=True)

    # Base margin (schema expects 'margin', not 'score_diff')
    out["margin"] = (out["score"] - out["opp_score"]).astype("float32")

    # Rolling margin (last 5 and 10 games, leakage-safe)
    out["margin_rolling_5"] = (
        out.groupby("team")["margin"]
        .shift(1)
        .rolling(5, min_periods=1)
        .mean()
        .astype("float32")
    )

    out["margin_rolling_10"] = (
        out.groupby("team")["margin"]
        .shift(1)
        .rolling(10, min_periods=1)
        .mean()
        .astype("float32")
    )

    return out
