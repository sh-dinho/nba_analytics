import streamlit as st
import pandas as pd

st.set_page_config(page_title="Weekly Summary", layout="wide")
st.title("🏆 Weekly Summary Report")

try:
    summary = pd.read_csv("results/weekly_summary.csv").iloc[0]

    st.subheader("Player of the Week")
    st.metric("Name", summary["player_of_week"])
    st.metric("PTS Change", f'{summary["player_pts_change"]:+.1f}')
    st.metric("REB Change", f'{summary["player_reb_change"]:+.1f}')
    st.metric("AST Change", f'{summary["player_ast_change"]:+.1f}')
    st.metric("Efficiency Change (TS%)", f'{summary["player_eff_change"]:+.3f}')

    st.subheader("Team of the Week")
    st.metric("Team", summary["team_of_week"])

except FileNotFoundError:
    st.error("Weekly summary not found. Run weekly_summary.py after updating trends.")