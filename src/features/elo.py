from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: ELO Features
# File: src/features/elo.py
# Author: Sadiq
#
# Description:
#     Leakage-safe ELO rating for team-game rows.
#     Uses canonical long-format columns:
#         - score
#         - opp_score
#     Produces:
#         - elo (pre-game)
#         - opp_elo (opponent pre-game)
# ============================================================

import pandas as pd


def _apply_elo(df: pd.DataFrame, k: float = 20.0) -> pd.Series:
    """
    Compute pre-game ELO for each team in chronological order.
    Uses canonical columns: score, opp_score.
    """
    elo_map = {}  # team -> current ELO
    results = []

    # Sort chronologically to avoid leakage
    df_sorted = df.sort_values("date")

    for row in df_sorted.itertuples():
        team = row.team
        opp = row.opponent

        # Current ratings (default 1500)
        elo_team = elo_map.get(team, 1500.0)
        elo_opp = elo_map.get(opp, 1500.0)

        # Pre-game ELO is the feature
        results.append(elo_team)

        # Update only if scores exist
        if pd.notna(row.score) and pd.notna(row.opp_score):
            actual = 1.0 if row.score > row.opp_score else (0.5 if row.score == row.opp_score else 0.0)
            expected = 1.0 / (1.0 + 10 ** ((elo_opp - elo_team) / 400))

            # Update team ELO
            elo_map[team] = elo_team + k * (actual - expected)

    return pd.Series(results, index=df_sorted.index)


def add_elo_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
        - elo: pre-game ELO for each team
        - opp_elo: opponent's pre-game ELO
    """
    out = df.copy()

    # Compute pre-game ELO
    out["elo"] = _apply_elo(out)

    # Merge opponent ELO
    opp_elo = (
        out[["game_id", "team", "elo"]]
        .rename(columns={"team": "opponent", "elo": "opp_elo"})
    )

    out = out.merge(opp_elo, on=["game_id", "opponent"], how="left")

    return out
