from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Opponent-Adjusted Features
# File: src/features/opponent_adjusted.py
# Author: Sadiq
#
# Description:
#     Opponent-adjusted rolling statistics for team-game data.
#     For each team, merges the opponent's rolling margin into
#     the current row using game_id to align matchups.
# ============================================================

import pandas as pd


WINDOWS = [5, 10]


def add_opponent_adjusted_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds opponent-adjusted rolling margin features.

    Required columns:
        - game_id
        - team
        - opponent
        - roll_margin_5 (optional)
        - roll_margin_10 (optional)
    """
    out = df.copy()

    for w in WINDOWS:
        src_col = f"roll_margin_{w}"
        opp_col = f"opp_roll_margin_{w}"

        if src_col not in out.columns:
            continue

        # Build opponent lookup table
        opp_df = (
            out[["game_id", "team", src_col]]
            .rename(columns={"team": "opponent", src_col: opp_col})
        )

        # Merge opponent rolling stats
        out = out.merge(opp_df, on=["game_id", "opponent"], how="left")

    return out