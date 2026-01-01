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
        - game_id
        - score
        - opponent_score
    """
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values(["team", "date", "game_id"])

    # Pre-calculate indicators (memory optimized)
    out["score_diff"] = (out["score"] - out["opponent_score"]).astype("float32")
    out["win_flag"] = (out["score"] > out["opponent_score"]).astype("float32")

    grouped = out.groupby("team")

    for w in WINDOWS:
        # Vectorized rolling: shift → rolling → mean
        out[f"roll_points_for_{w}"] = (
            grouped["score"].shift().rolling(w, min_periods=1).mean().astype("float32")
        )
        out[f"roll_points_against_{w}"] = (
            grouped["opponent_score"].shift().rolling(w, min_periods=1).mean().astype("float32")
        )
        out[f"roll_margin_{w}"] = (
            grouped["score_diff"].shift().rolling(w, min_periods=1).mean().astype("float32")
        )
        out[f"roll_win_rate_{w}"] = (
            grouped["win_flag"].shift().rolling(w, min_periods=1).mean().astype("float32")
        )

    return out