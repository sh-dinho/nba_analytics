# app/app.py
import streamlit as st
import pandas as pd
import logging
from config import configure_logging
from nba_analytics_core.db_module import init_db
from nba_analytics_core.predictor import predict_todays_games
from nba_analytics_core.simulate_ai_bankroll import simulate_ai_strategy

def main():
    configure_logging()
    st.title("NBA Analytics Dashboard")
    st.caption("Predictions and bankroll simulation")

    init_db()
    threshold = st.slider("Prediction threshold", 0.5, 0.8, 0.6, 0.01)
    strategy = st.selectbox("Bankroll strategy", ["flat", "kelly"])
    bankroll = st.number_input("Initial bankroll", min_value=100.0, value=1000.0, step=50.0)

    with st.spinner("Generating predictions..."):
        preds = predict_todays_games(threshold=threshold)
    st.subheader("Today's Predictions")
    st.dataframe(preds)

    with st.spinner("Running simulation..."):
        sim = simulate_ai_strategy(initial_bankroll=bankroll, strategy=strategy)
    st.subheader("Simulation Results")
    st.dataframe(sim[["game_id", "team", "decimal_odds", "prob", "stake", "pnl", "bankroll"]])

    kpis = getattr(sim, "kpis", {})
    st.metric("ROI", f"{kpis.get('roi', 0.0):.2%}")
    st.metric("Win Rate", f"{kpis.get('win_rate', 0.0):.2%}")
    st.metric("Max Drawdown", f"{kpis.get('max_drawdown', 0.0):.2f}")

if __name__ == "__main__":
    main()