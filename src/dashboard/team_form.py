# ============================================================
# 🏀 NBA Analytics v4
# Module: Team Form Charts
# File: src/dashboard/team_form.py
# Author: Sadiq
#
# Description:
#     Visualizes team form using ELO and rolling stats.
#     Includes:
#       - ELO trend
#       - Rolling margin
#       - Rolling win rate
#       - Optional: win/loss markers
# ============================================================

from __future__ import annotations

import pandas as pd
import streamlit as st
import altair as alt


def team_form_view(features_df: pd.DataFrame):
    st.subheader("📈 Team Form Charts")

    # ------------------------------------------------------------
    # Team selector
    # ------------------------------------------------------------
    teams = sorted(features_df["team"].unique())
    team = st.selectbox("Select team", teams)

    df = features_df[features_df["team"] == team].sort_values("date")

    # Ensure date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])

    # ------------------------------------------------------------
    # Optional win/loss markers
    # ------------------------------------------------------------
    show_markers = st.checkbox("Show win/loss markers on ELO chart", value=True)

    if "result" in df.columns:
        df["result_marker"] = df["result"].map({"W": 1, "L": 0})
    else:
        df["result_marker"] = None

    # ------------------------------------------------------------
    # ELO chart
    # ------------------------------------------------------------
    st.markdown("### 🧠 ELO Trend")

    base = alt.Chart(df).encode(x="date:T")

    elo_line = base.mark_line(color="#1f77b4").encode(
        y="elo:Q",
        tooltip=["date", "elo"],
    )

    if show_markers and df["result_marker"].notna().any():
        markers = base.mark_point(size=60).encode(
            y="elo:Q",
            color=alt.condition(
                alt.datum.result_marker == 1,
                alt.value("green"),
                alt.value("red"),
            ),
            tooltip=["date", "elo", "result"],
        )
        elo_chart = (elo_line + markers).properties(
            title=f"{team} — ELO Over Time", height=260
        )
    else:
        elo_chart = elo_line.properties(title=f"{team} — ELO Over Time", height=260)

    st.altair_chart(elo_chart, use_container_width=True)

    # ------------------------------------------------------------
    # Rolling margin
    # ------------------------------------------------------------
    if "roll_margin_10" in df.columns:
        st.markdown("### 📊 Rolling Margin (10 games)")
        margin_chart = (
            alt.Chart(df)
            .mark_line(color="orange")
            .encode(
                x="date:T",
                y="roll_margin_10:Q",
                tooltip=["date", "roll_margin_10"],
            )
            .properties(height=250)
        )
        st.altair_chart(margin_chart, use_container_width=True)
    else:
        st.info("Rolling margin column 'roll_margin_10' not found.")

    # ------------------------------------------------------------
    # Rolling win rate
    # ------------------------------------------------------------
    if "roll_win_rate_10" in df.columns:
        st.markdown("### 🟩 Rolling Win Rate (10 games)")
        win_chart = (
            alt.Chart(df)
            .mark_line(color="green")
            .encode(
                x="date:T",
                y="roll_win_rate_10:Q",
                tooltip=["date", "roll_win_rate_10"],
            )
            .properties(height=250)
        )
        st.altair_chart(win_chart, use_container_width=True)
    else:
        st.info("Rolling win rate column 'roll_win_rate_10' not found.")

    # ------------------------------------------------------------
    # Optional combined chart
    # ------------------------------------------------------------
    if st.checkbox("Show combined multi‑metric chart", value=False):
        st.markdown("### 📚 Combined Metrics")

        melt_cols = []
        if "elo" in df.columns:
            melt_cols.append("elo")
        if "roll_margin_10" in df.columns:
            melt_cols.append("roll_margin_10")
        if "roll_win_rate_10" in df.columns:
            melt_cols.append("roll_win_rate_10")

        if melt_cols:
            melted = df.melt(
                id_vars=["date"],
                value_vars=melt_cols,
                var_name="metric",
                value_name="value",
            )

            combined_chart = (
                alt.Chart(melted)
                .mark_line()
                .encode(
                    x="date:T",
                    y="value:Q",
                    color="metric:N",
                    tooltip=["date", "metric", "value"],
                )
                .properties(height=300)
            )
            st.altair_chart(combined_chart, use_container_width=True)
        else:
            st.info("No metrics available for combined chart.")
