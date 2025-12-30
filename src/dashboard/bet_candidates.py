# ============================================================
# 🏀 NBA Analytics v4
# Module: Bet Candidates View
# File: src/dashboard/bet_candidates.py
# Author: Sadiq
#
# Description:
#     Displays top betting opportunities based on model edges.
#     Requires combined predictions + market odds.
# ============================================================

from __future__ import annotations
import pandas as pd
import streamlit as st


def bet_candidates_view(combined: pd.DataFrame):
    st.subheader("🎯 Bet Candidates")

    # ------------------------------------------------------------
    # Validate required columns
    # ------------------------------------------------------------
    if "home_edge" not in combined.columns and "away_edge" not in combined.columns:
        st.info("No edge columns found. Add market odds ingestion to enable this view.")
        return

    # ------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------
    st.markdown("### Filters")

    min_edge = st.slider(
        "Minimum edge (%)",
        min_value=0.0,
        max_value=20.0,
        value=3.0,
        help="Minimum model edge required to consider a bet.",
    )

    min_prob = st.slider(
        "Minimum win probability (%)",
        min_value=0.0,
        max_value=100.0,
        value=40.0,
        help="Minimum model win probability required.",
    )

    df = combined.copy()

    # ------------------------------------------------------------
    # Convert edges to %
    # ------------------------------------------------------------
    if "home_edge" in df:
        df["home_edge_pct"] = df["home_edge"] * 100
    if "away_edge" in df:
        df["away_edge_pct"] = df["away_edge"] * 100

    # ------------------------------------------------------------
    # Candidate extraction
    # ------------------------------------------------------------
    candidates = []

    for _, row in df.iterrows():
        # Home side candidate
        if (
            "home_edge_pct" in row
            and row["home_edge_pct"] >= min_edge
            and "win_probability_home" in row
            and row["win_probability_home"] * 100 >= min_prob
        ):
            candidates.append(
                {
                    "game_id": row["game_id"],
                    "side": f"{row['home_team']} ML",
                    "edge (%)": round(row["home_edge_pct"], 2),
                    "prob (%)": round(row["win_probability_home"] * 100, 2),
                }
            )

        # Away side candidate
        if (
            "away_edge_pct" in row
            and row["away_edge_pct"] >= min_edge
            and "win_probability_away" in row
            and row["win_probability_away"] * 100 >= min_prob
        ):
            candidates.append(
                {
                    "game_id": row["game_id"],
                    "side": f"{row['away_team']} ML",
                    "edge (%)": round(row["away_edge_pct"], 2),
                    "prob (%)": round(row["win_probability_away"] * 100, 2),
                }
            )

    # ------------------------------------------------------------
    # Display results
    # ------------------------------------------------------------
    if not candidates:
        st.warning("No bet candidates meet the criteria.")
        return

    cand_df = pd.DataFrame(candidates).sort_values("edge (%)", ascending=False)

    st.dataframe(cand_df, use_container_width=True)
