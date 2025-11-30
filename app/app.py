import streamlit as st
import numpy as np
import logging

from models.train_models import train_models_cached
from app.visualizations.evaluation_plots import (
    plot_confusion_matrix_with_metrics,
    plot_regression_evaluation,
    plot_simulation_results
)
from simulate_bankroll import simulate_bankroll
from core.fetch_games import get_todays_games
from core.db_module import get_bet_history, init_db
from app.visualizations.visualizations import plot_bet_history
from app.visualizations.team_visualizations import plot_team_performance

logging.basicConfig(level=logging.INFO)
st.set_page_config(page_title="NBA Analytics Dashboard", layout="wide")
st.title("🏀 NBA Analytics Dashboard")

# Ensure DB is initialized
init_db()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Classification eval", "Regression eval", "Simulation", "Today's games", "Team performance", "DB Verification"
])

# --- Classification Evaluation ---
with tab1:
    st.subheader("Classification model evaluation")
    clf, reg, eval_class, eval_reg = train_models_cached("2025", ttl=600)
    if clf and eval_class:
        X_test_c, y_test_c, y_pred_c, y_proba_c, acc = eval_class
        plot_confusion_matrix_with_metrics(y_test_c, y_pred_c, labels=["Loss", "Win"], normalize=True)
        st.write(f"Model Accuracy: {acc:.2%}")
    else:
        st.info("Train models first or check data availability.")

# --- Regression Evaluation ---
with tab2:
    st.subheader("Regression model evaluation")
    if reg and eval_reg:
        y_test_r, y_pred_r, r2 = eval_reg
        plot_regression_evaluation(y_test_r, y_pred_r)
        st.write(f"R²: {r2:.3f}")
    else:
        st.info("Train models first or check data availability.")

# --- Simulation ---
with tab3:
    st.subheader("Bankroll simulation")
    colA, colB, colC, colD = st.columns(4)
    with colA:
        starting_bankroll = st.number_input("Starting bankroll", value=1000.0, min_value=0.0, step=50.0)
    with colB:
        odds = st.number_input("Decimal odds", value=2.0, min_value=1.01, step=0.05)
    with colC:
        games = st.number_input("Games", value=50, min_value=1, step=1)
    with colD:
        method = st.selectbox("Staking method", options=["kelly", "flat"], index=0)

    conf_factor = st.slider("Confidence factor", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    min_stake = st.number_input("Minimum stake", value=10.0, min_value=0.0, step=5.0)
    tx_fee = st.slider("Transaction fee (winnings)", min_value=0.0, max_value=0.10, value=0.0, step=0.005)
    seed = st.number_input("Random seed (optional)", value=42, min_value=0, step=1)

    probs = np.random.uniform(0.4, 0.6, games)
    history, final_bankroll, roi, win_rate, max_drawdown = simulate_bankroll(
        probs=probs,
        odds=odds,
        starting_bankroll=starting_bankroll,
        method=method,
        confidence_factor=conf_factor,
        min_stake=min_stake,
        games=games,
        transaction_fee=tx_fee,
        seed=seed
    )
    plot_simulation_results(history, starting_bankroll, final_bankroll, roi, win_rate, max_drawdown)

# --- Today's Games ---
with tab4:
    st.subheader("Today's NBA games")
    df_today = get_todays_games()
    if df_today.empty:
        st.warning("No games found for today in the database.")
    else:
        st.dataframe(df_today[["home_team", "away_team", "date", "home_score", "away_score", "winner"]])

    st.subheader("Tracked bets history")
    df_bets = get_bet_history(limit=100)
    plot_bet_history(df_bets)

# --- Team Performance ---
with tab5:
    st.subheader("Team performance by season")
    season_opt = st.selectbox("Select season", options=["All", 2025, 2024, 2023])
    season = None if season_opt == "All" else season_opt
    plot_team_performance(season)

# --- DB Verification ---
with tab6:
    st.subheader("Database Verification")
    import sqlite3, yaml, pandas as pd
    with open("config.yaml") as f:
        CONFIG = yaml.safe_load(f)
    DB_PATH = CONFIG["database"]["path"]
    with sqlite3.connect(DB_PATH) as con:
        schema = con.execute("PRAGMA table_info(nba_games);").fetchall()
        st.write("### Schema for nba_games")
        st.table(schema)
        count = con.execute("SELECT COUNT(*) FROM nba_games;").fetchone()[0]
        st.write(f"Total rows: {count}")
        df_sample = pd.read_sql("SELECT * FROM nba_games LIMIT 10;", con)
        st.write("### Sample rows")
        st.dataframe(df_sample)