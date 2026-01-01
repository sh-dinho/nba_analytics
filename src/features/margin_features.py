from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Margin Features
# File: src/features/margin_features.py
# Author: Sadiq
#
# Description:
#     Adds scoring margin features for team-game rows:
#     - score_diff
#     - margin_last5
#     - margin_last10
# ============================================================

import pandas as pd


def add_margin_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds scoring margin features.

    Required columns:
        - team
        - date
        - game_id
        - score
        - opponent_score
    """
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values(["team", "date", "game_id"])

    # Base margin
    out["score_diff"] = (out["score"] - out["opponent_score"]).astype("float32")

    # Rolling margin (last 5 and 10 games)
    out["margin_last5"] = (
        out.groupby("team")["score_diff"]
        .shift()
        .rolling(5, min_periods=1)
        .mean()
        .astype("float32")
    )

    out["margin_last10"] = (
        out.groupby("team")["score_diff"]
        .shift()
        .rolling(10, min_periods=1)
        .mean()
        .astype("float32")
    )

    return out