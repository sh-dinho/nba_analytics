from __future__ import annotations
import pandas as pd

# ============================================================
# 🏀 NBA Analytics
# Module: Opponent-Adjusted Features
# File: src/features/opponent_adjusted.py
# Author: Sadiq
#
# Description:
#     Adds opponent-adjusted rolling statistics by aligning
#     opponent metrics using game_id + opponent.
# ============================================================


def add_opponent_adjusted_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds opponent-adjusted rolling features:
        • opp_margin_rolling_5
        • opp_margin_rolling_10
        • opp_win_pct_last10

    Uses game_id + opponent to align the opponent’s rolling stats.
    """
    out = df.copy()

    # --------------------------------------------------------
    # Opponent margin rolling (5 & 10 games)
    # --------------------------------------------------------
    for w in [5, 10]:
        margin_col = f"margin_rolling_{w}"
        opp_col = f"opp_margin_rolling_{w}"

        if margin_col not in out.columns:
            continue  # skip if rolling margin not computed yet

        opp_lookup = (
            out[["game_id", "team", margin_col]]
            .rename(columns={"team": "opponent", margin_col: opp_col})
        )

        out = out.merge(
            opp_lookup,
            on=["game_id", "opponent"],
            how="left",
            suffixes=("", "_dup"),
        )

        # Remove duplicate columns created by merge
        dup_cols = [c for c in out.columns if c.endswith("_dup")]
        if dup_cols:
            out.drop(columns=dup_cols, inplace=True)

    # --------------------------------------------------------
    # Opponent win% last 10 games
    # --------------------------------------------------------
    if "team_win_pct_last10" in out.columns:
        opp_win_lookup = (
            out[["game_id", "team", "team_win_pct_last10"]]
            .rename(
                columns={
                    "team": "opponent",
                    "team_win_pct_last10": "opp_win_pct_last10",
                }
            )
        )

        out = out.merge(
            opp_win_lookup,
            on=["game_id", "opponent"],
            how="left",
            suffixes=("", "_dup"),
        )

        dup_cols = [c for c in out.columns if c.endswith("_dup")]
        if dup_cols:
            out.drop(columns=dup_cols, inplace=True)

    return out