# app/pages/ai_predictions.py (or inside your main app under the "AI Predictions" tab)
import streamlit as st
import pandas as pd
from app.predict_pipeline import generate_predictions, build_bets_from_predictions
from scripts.simulate_bankroll import simulate_bankroll

def render_ai_predictions_page():
    st.subheader("AI Predictions and Bankroll Simulation")

    threshold = st.sidebar.slider("Strong pick threshold", 0.5, 0.8, 0.6, 0.01)
    strategy = st.sidebar.selectbox("Betting strategy", ["kelly", "flat"])
    max_fraction = st.sidebar.slider("Max Kelly fraction per bet", 0.01, 0.10, 0.05, 0.01)

    preds = generate_predictions(threshold=threshold)
    if preds.empty:
        st.warning("No predictions available.")
        return

    preds["Strong Pick"] = preds["strong_pick"].apply(lambda x: "✅" if x == 1 else "—")
    st.dataframe(
        preds[["home_team", "away_team", "pred_home_win_prob", "pred_total_points", "home_decimal_odds", "Strong Pick"]],
        use_container_width=True
    )

    # Build bets and simulate
    bets = build_bets_from_predictions(preds, threshold=threshold)
    if bets.empty:
        st.info("No qualifying bets at this threshold.")
        return

    sim = simulate_bankroll(bets, strategy=strategy, max_fraction=max_fraction)
    st.subheader("Bankroll trajectory")
    st.line_chart(sim["bankroll"])

    # Download predictions and bets
    col1, col2 = st.columns(2)
    with col1:
        csv_preds = preds.to_csv(index=False).encode("utf-8")
        st.download_button("Download predictions CSV", csv_preds, "predictions.csv", "text/csv")
    with col2:
        csv_bets = bets.to_csv(index=False).encode("utf-8")
        st.download_button("Download bets CSV", csv_bets, "bets.csv", "text/csv")