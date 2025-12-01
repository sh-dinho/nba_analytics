# Path: app/app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import logging

try:
    from nba_analytics_core.data import fetch_today_games, build_team_stats, build_matchup_features
    from nba_analytics_core.player_data import fetch_player_season_stats, build_player_leaderboards
    from nba_analytics_core.team_strength import championship_probabilities
except ImportError as e:
    st.error(f"Failed to import core modules: {e}")

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="NBA Analytics Unified Dashboard", layout="wide")

# --- Sidebar Navigation ---
st.sidebar.title("🏀 NBA Analytics Dashboard")
page = st.sidebar.radio(
    "Navigate",
    [
        "Daily Predictions",
        "Over/Under Forecasts",
        "Player Tracker",
        "Player Trends",
        "Weekly Summary",
        "Awards Forecasts"
    ],
    key="nav_radio"   # ✅ unique key for navigation
)

# --- Daily Predictions ---
if page == "Daily Predictions":
    st.title("📅 Daily Predictions")
    st.caption("Win probabilities for today's games using the trained model.")

    try:
        from app.predict_pipeline import generate_today_predictions
        df = generate_today_predictions()
        if df.empty:
            st.info("No games found today or model not trained yet.")
        else:
            st.subheader("Predictions")
            st.dataframe(df, use_container_width=True)

            st.subheader("Top picks")
            threshold = st.sidebar.slider(
                "Probability threshold", 0.5, 0.9, 0.65, 0.01, key="threshold_slider"
            )  # ✅ unique key
            st.dataframe(df[df["home_win_prob"] >= threshold], use_container_width=True)
    except Exception as e:
        st.error(f"Error generating predictions: {e}")

# --- Over/Under Forecasts ---
elif page == "Over/Under Forecasts":
    st.title("📊 Over/Under Forecasts")
    st.caption("Totals predictions vs your selected line. This is model-based, not a betting tool.")

    try:
        from app.predict_pipeline import generate_today_predictions_with_totals
        line = st.sidebar.number_input(
            "Points line", min_value=180.0, max_value=260.0, value=220.0, step=0.5, key="ou_line_input"
        )  # ✅ unique key
        df = generate_today_predictions_with_totals(line=line)

        if df.empty:
            st.info("No games found today or O/U model not trained yet.")
        else:
            st.subheader("Totals predictions")
            st.dataframe(df[["date","home_team","away_team","line","prob_over","prob_under"]], use_container_width=True)

            st.subheader("Over vs Under probability")
            labels = df["home_team"] + " vs " + df["away_team"]
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(labels, df["prob_over"], label="Over", color="steelblue")
            ax.bar(labels, df["prob_under"], bottom=df["prob_over"], label="Under", color="orange")
            ax.set_ylabel("Probability")
            ax.legend()
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig)
    except Exception as e:
        st.error(f"Error generating O/U predictions: {e}")

# --- Player Tracker ---
elif page == "Player Tracker":
    st.title("⭐ Player Tracker: Top Performers & Players to Watch")
    season = st.sidebar.text_input("Season", "2025-26", key="season_input")  # ✅ unique key
    mode = st.sidebar.selectbox("Per-mode", ["PerGame", "Totals"], index=0, key="mode_select")  # ✅ unique key
    top_n = st.sidebar.slider("Top N Players", 5, 20, 10, key="topn_slider")  # ✅ unique key

    try:
        df = fetch_player_season_stats(season=season, per_mode=mode)
        boards = build_player_leaderboards(df, top_n=top_n)
    except Exception as e:
        st.error(f"Failed to fetch player stats: {e}")
        df = pd.DataFrame()
        boards = {}

    if not df.empty:
        st.subheader(f"Top {top_n} Scorers")
        if "Scoring (PTS)" in boards:
            st.dataframe(boards["Scoring (PTS)"], use_container_width=True)

        st.subheader("Players to Watch")
        watch_list = df[df["PLAYER_NAME"].isin([
            "Victor Wembanyama","Chet Holmgren","Shaedon Sharpe","Jalen Williams","Alperen Sengun","Cooper Flagg"
        ])]
        st.dataframe(watch_list[["PLAYER_NAME","TEAM_ABBREVIATION","PTS","REB","AST","TS_PCT"]], use_container_width=True)

# --- Player Trends ---
elif page == "Player Trends":
    st.title("📈 Weekly Player Trends")
    try:
        df = pd.read_csv("results/player_trends.csv")
        st.subheader("Top Risers")
        st.dataframe(df[df["trend"]=="Rising"].sort_values("PTS_change",ascending=False).head(10))
        st.subheader("Top Fallers")
        st.dataframe(df[df["trend"]=="Falling"].sort_values("PTS_change").head(10))
    except FileNotFoundError:
        st.error("Trend data not found. Run weekly update.")

# --- Weekly Summary ---
elif page == "Weekly Summary":
    st.title("🏆 Weekly Summary Report")
    try:
        summary = pd.read_csv("results/weekly_summary.csv").iloc[0]
        st.metric("Player of the Week", summary["player_of_week"])
        st.metric("Team of the Week", summary["team_of_week"])
    except FileNotFoundError:
        st.error("Weekly summary not found. Run weekly_summary.py.")

# --- Awards Forecasts ---
elif page == "Awards Forecasts":
    st.title("🏆 Awards Forecasts")
    try:
        champ = pd.read_csv("results/championship_probs.csv")
        st.subheader("Championship Probabilities")
        st.dataframe(champ, use_container_width=True)

        mvp = pd.read_csv("results/mvp_probs.csv")
        st.subheader("MVP Probabilities")
        st.dataframe(mvp.head(20), use_container_width=True)
    except FileNotFoundError:
        st.error("Awards data not found. Run train_awards.py.")