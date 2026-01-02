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
        - opp_score
    """
    out = df.copy()

    # Validate required columns
    required = {"team", "date", "score", "opp_score"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"add_form_features missing columns: {missing}")

    # Ensure datetime
    out["date"] = pd.to_datetime(out["date"])

    # Sort for rolling operations
    out = out.sort_values(["team", "date"]).reset_index(drop=True)

    # Compute score differential
    out["score_diff"] = out["score"] - out["opp_score"]

    # Rolling average margin over last 3 games (excluding current game)
    out["form_last3"] = (
        out.groupby("team")["score_diff"]
        .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    )

    return out
