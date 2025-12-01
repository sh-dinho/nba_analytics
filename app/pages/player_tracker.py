import streamlit as st
import pandas as pd
from nba_analytics_core.player_data import fetch_player_season_stats, build_player_leaderboards

st.set_page_config(page_title="Player Tracker", layout="wide")
st.title("⭐ Player Tracker: Top Performers & Players to Watch")

season = st.sidebar.text_input("Season", "2025-26")
mode = st.sidebar.selectbox("Per-mode", ["PerGame", "Totals"], index=0)
top_n = 10  # fixed to top 10

# Fetch player stats
df = fetch_player_season_stats(season=season, per_mode=mode)
boards = build_player_leaderboards(df, top_n=top_n)

st.subheader("Eastern Conference – Top 10")
east = df[df["TEAM_ABBREVIATION"].isin(["BOS","NYK","PHI","MIL","MIA","ATL","CHI","CLE","DET","TOR"])]
st.dataframe(east.sort_values("PTS", ascending=False).head(top_n), use_container_width=True)

st.subheader("Western Conference – Top 10")
west = df[df["TEAM_ABBREVIATION"].isin(["LAL","GSW","DEN","MIN","OKC","SAS","PHX","UTA","HOU","DAL","POR"])]
st.dataframe(west.sort_values("PTS", ascending=False).head(top_n), use_container_width=True)

st.subheader("Players to Watch (Breakout Candidates)")
# Example: highlight young players or rising stars
watch_list = df[df["PLAYER_NAME"].isin([
    "Victor Wembanyama","Chet Holmgren","Shaedon Sharpe","Jalen Williams","Alperen Sengun","Cooper Flagg"
])]
st.dataframe(watch_list[["PLAYER_NAME","TEAM_ABBREVIATION","PTS","REB","AST","TS_PCT"]], use_container_width=True)