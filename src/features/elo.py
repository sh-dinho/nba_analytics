from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: ELO Features
# File: src/features/elo.py
# Author: Sadiq
#
# Description:
#     ELO-style team rating for team-game rows, computed in a
#     leakage-safe manner. Also derives opponent ELO by merging
#     within each game_id.
# ============================================================

import pandas as pd


def _apply_elo_optimized(df: pd.DataFrame, k: float = 20.0) -> pd.Series:
    """
    Tight loop for ELO calculation to minimize overhead.
    Computes pre-game ELO for each team in chronological order.
    """
    elo_map = {}  # Team -> Current ELO
    results = []

    # Iterate over sorted records to prevent data leakage
    for row in df.itertuples():
        t, o = row.team, row.opponent
        e_t = elo_map.get(t, 1500.0)
        e_o = elo_map.get(o, 1500.0)

        # Record pre-game ELO
        results.append(e_t)

        # Update ELO only if scores are available
        if pd.notna(row.score) and pd.notna(row.opponent_score):
            actual = (
                1.0 if row.score > row.opponent_score
                else 0.5 if row.score == row.opponent_score
                else 0.0
            )
            expected = 1.0 / (1.0 + 10.0 ** ((e_o - e_t) / 400.0))
            elo_map[t] = e_t + k * (actual - expected)

    return pd.Series(results, index=df.index, dtype="float32")


def add_elo_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds ELO and opponent ELO to the long-format team-game dataframe.

    Required columns:
        - team
        - opponent
        - date
        - game_id
        - score
        - opponent_score
    """
    out = df.sort_values(["date", "game_id", "team"]).copy()
    out["elo"] = _apply_elo_optimized(out)

    # Vectorized opponent ELO alignment
    opp_elo = (
        out[["game_id", "team", "elo"]]
        .rename(columns={"team": "opponent", "elo": "opp_elo"})
    )

    return out.merge(opp_elo, on=["game_id", "opponent"], how="left")