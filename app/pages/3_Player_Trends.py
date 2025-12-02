# ============================================================
# File: 3_Weekly_Player_Trends.py
# Path: <project_root>/pages/3_Weekly_Player_Trends.py
#
# Description:
#   Streamlit page to track weekly player performance trends
#   across multiple metrics (points, assists, rebounds). Supports
#   filtering by player or team, interactive charts, metrics, and
#   CSV export.
# ============================================================

import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime

st.title("📈 Weekly Player Trends")
st.caption("Track player performance across weeks.")

try:
    df_trends = pd.read_csv("results/player_trends.csv")

    if df_trends.empty:
        st.info("No player trends data available. Run the pipeline first.")
    else:
        # Show raw table
        st.subheader("Raw Trends Data")
        st.dataframe(df_trends, use_container_width=True)

        # Sidebar filters
        players = sorted(df_trends["PLAYER_NAME"].unique())
        teams = sorted(df_trends["TEAM_ABBREVIATION"].unique())
        selected_player = st.sidebar.selectbox("Select Player", ["All"] + players)
        selected_team = st.sidebar.selectbox("Select Team", ["All"] + teams)

        filtered = df_trends.copy()
        if selected_player != "All":
            filtered = filtered[filtered["PLAYER_NAME"] == selected_player]
        if selected_team != "All":
            filtered = filtered[filtered["TEAM_ABBREVIATION"] == selected_team]

        # Metrics
        st.subheader("Key Metrics")
        st.metric("Players Tracked", df_trends["PLAYER_NAME"].nunique())
        st.metric("Teams Covered", df_trends["TEAM_ABBREVIATION"].nunique())
        st.metric("Weeks", df_trends["week"].nunique())

        # Trend charts
        if "PTS" in filtered.columns:
            st.subheader("Points Trend")
            chart_pts = alt.Chart(filtered).mark_line(point=True).encode(
                x="week:O",
                y="PTS:Q",
                color="PLAYER_NAME:N",
                tooltip=["PLAYER_NAME","TEAM_ABBREVIATION","week","PTS"]
            ).interactive()
            st.altair_chart(chart_pts, use_container_width=True)

        if "AST" in filtered.columns:
            st.subheader("Assists Trend")
            chart_ast = alt.Chart(filtered).mark_line(point=True).encode(
                x="week:O",
                y="AST:Q",
                color="PLAYER_NAME:N",
                tooltip=["PLAYER_NAME","TEAM_ABBREVIATION","week","AST"]
            ).interactive()
            st.altair_chart(chart_ast, use_container_width=True)

        if "REB" in filtered.columns:
            st.subheader("Rebounds Trend")
            chart_reb = alt.Chart(filtered).mark_line(point=True).encode(
                x="week:O",
                y="REB:Q",
                color="PLAYER_NAME:N",
                tooltip=["PLAYER_NAME","TEAM_ABBREVIATION","week","REB"]
            ).interactive()
            st.altair_chart(chart_reb, use_container_width=True)

        # Export option
        export_name = f"player_trends_{datetime.now().date()}.csv"
        st.download_button(
            label="Download Player Trends as CSV",
            data=filtered.to_csv(index=False),
            file_name=export_name,
            mime="text/csv"
        )

except Exception as e:
    st.error(f"Error loading player trends: {e}")
