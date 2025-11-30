import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from team_performance import get_team_performance

def plot_team_performance(season: int = None):
    """
    Plot team performance metrics (wins, losses, win percentage, points scored/allowed).
    If season is None, show all seasons combined.
    """
    df = get_team_performance(season)
    if df is None or df.empty:
        st.warning("No team performance data available.")
        return

    st.write("### Team Performance Table")
    st.dataframe(df)

    # --- Wins vs Losses ---
    fig, ax = plt.subplots(figsize=(10, 6))
    df_sorted = df.sort_values("win_percentage", ascending=False)
    sns.barplot(x="team_name", y="win_percentage", data=df_sorted, ax=ax, palette="Blues_d")
    ax.set_title(f"Win Percentage by Team ({season if season else 'All Seasons'})")
    ax.set_ylabel("Win %")
    ax.set_xlabel("Team")
    ax.tick_params(axis="x", rotation=90)
    st.pyplot(fig)

    # --- Points scored vs allowed ---
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    df_points = df.sort_values("points_scored", ascending=False)
    ax2.bar(df_points["team_name"], df_points["points_scored"], label="Points Scored", color="green")
    ax2.bar(df_points["team_name"], df_points["points_allowed"], label="Points Allowed", color="red", alpha=0.6)
    ax2.set_title(f"Points Scored vs Allowed ({season if season else 'All Seasons'})")
    ax2.set_ylabel("Points")
    ax2.set_xlabel("Team")
    ax2.tick_params(axis="x", rotation=90)
    ax2.legend()
    st.pyplot(fig2)