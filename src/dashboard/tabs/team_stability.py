# ============================================================
# 🏀 NBA Analytics v3
# Module: Dashboard — Team Stability (Avoid / Watch)
# File: src/dashboard/tabs/team_stability.py
# Author: Sadiq
#
# Description:
#     Visualizes team-level stability metrics:
#       - ROI per team
#       - Volatility
#       - Stability score
#       - Teams to avoid
#       - Teams to watch
#
#     Powered by TeamStabilityEngine.
# ============================================================

from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

from src.analytics.team_stability import TeamStabilityConfig, TeamStabilityEngine


def _plot_roi_vs_stability(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df["stability_score"], df["roi"], alpha=0.7)

    for _, row in df.iterrows():
        ax.annotate(
            row["team"],
            (row["stability_score"], row["roi"]),
            fontsize=8,
            alpha=0.7,
        )

    ax.set_xlabel("Stability Score")
    ax.set_ylabel("ROI")
    ax.set_title("Team ROI vs Stability Score")
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    fig.tight_layout()
    return fig


def render_team_stability_tab():
    st.header("Teams to Avoid & Watch")

    st.markdown(
        "_This view uses historical bets and predictions to identify "
        "which teams are stable, predictable, and profitable vs which "
        "teams are volatile and dangerous._"
    )

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Backtest start date (optional)", key="stab_start")
    with col2:
        end_date = st.date_input("Backtest end date (optional)", key="stab_end")

    col3, col4 = st.columns(2)
    with col3:
        min_bets = st.number_input("Minimum bets per team", value=20, min_value=1)
    with col4:
        avoid_roi_threshold = st.number_input("Avoid if ROI ≤", value=-0.05, step=0.01)
    watch_roi_threshold = st.number_input("Watch if ROI ≥", value=0.05, step=0.01)

    if st.button("Compute team stability"):
        cfg = TeamStabilityConfig(
            start_date=start_date.isoformat() if start_date else None,
            end_date=end_date.isoformat() if end_date else None,
            min_bets=min_bets,
            avoid_roi_threshold=avoid_roi_threshold,
            watch_roi_threshold=watch_roi_threshold,
        )
        engine = TeamStabilityEngine(cfg)
        res = engine.run()

        if res.teams.empty:
            st.warning(
                "No data available to compute team stability. Check backtest coverage."
            )
            return

        st.subheader("Per-team metrics")
        st.dataframe(
            res.teams.sort_values("stability_score", ascending=False),
            use_container_width=True,
        )

        st.subheader("Teams to Watch (high stability + positive ROI)")
        if res.teams_to_watch.empty:
            st.info("No teams currently meet the 'watch' criteria.")
        else:
            st.dataframe(
                res.teams_to_watch[
                    ["team", "stability_score", "roi", "num_bets", "hit_rate"]
                ].sort_values("stability_score", ascending=False),
                use_container_width=True,
            )

        st.subheader("Teams to Avoid (negative ROI / unstable)")
        if res.teams_to_avoid.empty:
            st.info("No teams currently meet the 'avoid' criteria.")
        else:
            st.dataframe(
                res.teams_to_avoid[
                    ["team", "stability_score", "roi", "num_bets", "hit_rate"]
                ].sort_values("stability_score", ascending=True),
                use_container_width=True,
            )

        st.subheader("ROI vs Stability")
        fig = _plot_roi_vs_stability(res.teams)
        st.pyplot(fig, clear_figure=True)
