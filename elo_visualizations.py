# elo_visualizations.py
import streamlit as st
import matplotlib.pyplot as plt
from data import fetch_historical_games, engineer_features

def plot_team_elo(season, team_name):
    df_games = fetch_historical_games(season)
    df_feat = engineer_features(df_games)
    if df_feat.empty:
        st.info("No data available.")
        return

    # Extract Elo progression for selected team
    elo = []
    dates = []
    for _, row in df_games.iterrows():
        if row["home_team"] == team_name:
            elo.append(row.get("home_elo", None))
            dates.append(row["date"])
        elif row["away_team"] == team_name:
            elo.append(row.get("away_elo", None))
            dates.append(row["date"])

    if not elo:
        st.warning("No games found for this team in the selected season.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates, elo, marker="o", linestyle="-", color="blue")
    ax.set_title(f"Elo progression for {team_name} ({season})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Elo rating")
    plt.xticks(rotation=45)
    st.pyplot(fig)