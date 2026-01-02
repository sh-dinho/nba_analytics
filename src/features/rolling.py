from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Rolling Features
# File: src/features/rolling.py
# Author: Sadiq
#
# Description:
#     Leakage-safe rolling statistics for team-game data:
#     points for/against, margin, and win rate over multiple
#     window sizes (5, 10, 20 games).
# ============================================================

import pandas as pd

WINDOWS = [5, 10, 20]


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds rolling points, margin, and win-rate features.

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
        raise ValueError(f"add_rolling_features missing columns: {missing}")

    # Ensure datetime
    out["date"] = pd.to_datetime(out["date"])

    # Sort for rolling operations
    out = out.sort_values(["team", "date"]).reset_index(drop=True)

    # Pre-calculate indicators
    out["margin"] = (out["score"] - out["opp_score"]).astype("float32")
    out["win_flag"] = (out["score"] > out["opp_score"]).astype("float32")

    grouped = out.groupby("team")

    for w in WINDOWS:
        # Rolling points for
        out[f"points_for_rolling_{w}"] = (
            grouped["score"]
            .shift(1)
            .rolling(w, min_periods=1)
            .mean()
            .astype("float32")
        )

        # Rolling points against
        out[f"points_against_rolling_{w}"] = (
            grouped["opp_score"]
            .shift(1)
            .rolling(w, min_periods=1)
            .mean()
            .astype("float32")
        )

        # Rolling margin
        out[f"margin_rolling_{w}"] = (
            grouped["margin"]
            .shift(1)
            .rolling(w, min_periods=1)
            .mean()
            .astype("float32")
        )

        # Rolling win rate
        out[f"win_rolling_{w}"] = (
            grouped["win_flag"]
            .shift(1)
            .rolling(w, min_periods=1)
            .mean()
            .astype("float32")
        )

    # Clean up temporary columns
    out = out.drop(columns=["win_flag"])

    return out
