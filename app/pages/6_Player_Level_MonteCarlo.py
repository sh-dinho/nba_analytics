import streamlit as st
import pandas as pd

st.title("👤 Player-Level Monte Carlo")

try:
    df_trends = pd.read_csv("results/player_trends.csv")
    player_choice = st.selectbox("Select Player", df_trends["PLAYER_NAME"].unique())
    team_abbr = df_trends.loc[df_trends["PLAYER_NAME"] == player_choice, "TEAM_ABBREVIATION"].head(1)
    if not team_abbr.empty:
        team = team_abbr.iloc[0]
        df_picks = pd.read_csv("results/picks.csv")
        picks_for_player = df_picks[(df_picks["home_team"] == team) | (df_picks["away_team"] == team)]
        st.dataframe(picks_for_player, use_container_width=True)
    else:
        st.info("No team mapping found for selected player.")
except Exception as e:
    st.error(f"Error loading player-level Monte Carlo: {e}")