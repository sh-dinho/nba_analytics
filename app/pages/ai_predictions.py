# app/pages/ai_predictions.py (or inside your main app under the "AI Predictions" tab)
import streamlit as st
import pandas as pd
from app.predict_pipeline import generate_predictions, build_bets_from_predictions
from scripts.simulate_bankroll import simulate_bankroll

def render_ai_predictions_page():
    st.subheader("AI Predictions and Bankroll Simulation")
    
    # Sidebar remains the same
    threshold = st.sidebar.slider("Strong pick threshold (Prob)", 0.5, 0.8, 0.6, 0.01)
    strategy = st.sidebar.selectbox("Betting strategy", ["kelly", "flat"])
    max_fraction = st.sidebar.slider("Max Kelly fraction per bet", 0.01, 0.10, 0.05, 0.01)

    # --- Prediction Generation ---
    preds = generate_predictions(threshold=threshold)
    if preds.empty:
        st.warning("No predictions available.")
        return

    # Add EV column to display
    preds["Strong Pick"] = preds["strong_pick"].apply(lambda x: "✅" if x == 1 else "—")
    st.dataframe(
        preds[[
            "home_team", 
            "away_team", 
            "pred_home_win_prob", 
            "home_decimal_odds", 
            "home_ev", # NEW: Display EV
            "pred_total_points", 
            "Strong Pick"
        ]],
        use_container_width=True,
        column_config={
            "home_ev": st.column_config.NumberColumn("EV", format="%.4f") # Format EV for readability
        }
    )

    # --- Build bets and simulate (Now based on Prob AND Positive EV) ---
    bets = build_bets_from_predictions(preds, threshold=threshold)
    if bets.empty:
        st.info(f"No qualifying bets found with a probability >= {threshold} AND positive Expected Value.")
        return

    # Simulation and Artifacts remain the same
    st.subheader(f"Qualifying Bets ({len(bets)} Found)")
    st.dataframe(
        bets[["team", "decimal_odds", "prob", "ev"]],
        use_container_width=True,
        column_config={
            "prob": st.column_config.NumberColumn("Model Prob", format="%.3f"),
            "ev": st.column_config.NumberColumn("EV", format="%.4f")
        }
    )
    
    sim = simulate_bankroll(bets, strategy=strategy, max_fraction=max_fraction)
    st.subheader("Bankroll Trajectory")
    st.line_chart(sim["bankroll"])
    # 

    # Download predictions and bets
    col1, col2 = st.columns(2)
    with col1:
        csv_preds = preds.to_csv(index=False).encode("utf-8")
        st.download_button("Download Predictions CSV", csv_preds, "predictions.csv", "text/csv")
    with col2:
        csv_bets = bets.to_csv(index=False).encode("utf-8")
        st.download_button("Download Bets CSV", csv_bets, "bets.csv", "text/csv")