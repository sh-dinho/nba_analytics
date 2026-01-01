from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Team Form Features
# File: src/features/form.py
# Author: Sadiq
#
# Description:
#     Team form metrics over recent games, such as average
#     score differential over the last 3 games.
# ============================================================

import pandas as pd


def add_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds team form metrics (e.g., last 3 games average margin).

    Required columns:
        - team
        - date
        - score
        - opponent_score
    """
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values(["team", "date", "game_id"])

    # Ensure score_diff exists
    if "score_diff" not in out.columns:
        out["score_diff"] = out["score"] - out["opponent_score"]

    # Rolling average margin over last 3 games (excluding current game)
    out["form_last3"] = (
        out.groupby("team")["score_diff"]
        .transform(lambda s: s.shift().rolling(3, min_periods=1).mean())
    )

    return out