from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v5.0
# Name: Simulation Lab
# File: src/app/pages/14_Simulation_Lab.py
# Purpose: Monte Carlo simulation of bankroll paths using
#          historical bet outcomes and simple strategies.
# ============================================================

from datetime import date
import numpy as np
import streamlit as st

from src.app.ui.header import render_header
from src.app.ui.page_state import set_active_page
from src.app.ui.navbar import render_navbar
from src.app.engines.bet_tracker import load_bets, simulate_bankroll

st.set_page_config(page_title="Simulation Lab", page_icon="🧪", layout="wide")

render_header()
set_active_page("Simulation Lab")
render_navbar()

st.title("🧪 Simulation Lab")

df = load_bets()
resolved = df[df["result"].isin(["win", "loss"])].copy()

if resolved.empty:
    st.info("You need some resolved bets in the Bet Tracker to run simulations.")
    st.stop()

st.markdown("### Historical performance snapshot")
st.write(f"Resolved bets: {len(resolved)}")
st.write(f"Mean payout per bet: {resolved['payout'].mean():.2f}")
st.write(f"Std of payout per bet: {resolved['payout'].std():.2f}")

st.markdown("---")
st.markdown("### Simulation configuration")

col1, col2, col3 = st.columns(3)
with col1:
    initial_bankroll = st.number_input("Initial bankroll", value=1000.0, min_value=0.0)
with col2:
    horizon = st.number_input(
        "Number of bets to simulate", value=200, min_value=10, max_value=2000
    )
with col3:
    n_sims = st.number_input(
        "Number of simulations", value=3000, min_value=100, max_value=10000, step=100
    )

strategy = st.selectbox("Staking strategy", ["Flat stake"])

flat_stake = st.number_input("Flat stake per bet", value=50.0, min_value=0.0)

st.markdown("---")
st.markdown("### Run simulation")

if st.button("Run simulation"):
    payouts = resolved["payout"].to_numpy(dtype=float)
    stakes = resolved["stake"].to_numpy(dtype=float)

    median_stake = np.median(stakes) if len(stakes) > 0 else 1.0
    scale = flat_stake / median_stake if median_stake > 0 else 1.0
    payouts_scaled = payouts * scale

    sims = simulate_bankroll(
        payouts=payouts_scaled,
        initial_bankroll=initial_bankroll,
        horizon=int(horizon),
        n_sims=int(n_sims),
    )

    mean_path = sims.mean(axis=0)
    p10 = np.percentile(sims, 10, axis=0)
    p90 = np.percentile(sims, 90, axis=0)

    st.markdown("### Bankroll paths (mean and 10–90% band)")
    st.line_chart(
        {
            "Mean": mean_path,
            "10th percentile": p10,
            "90th percentile": p90,
        }
    )

    risk_of_ruin = float((sims[:, -1] <= 0).mean())
    st.write(f"**Risk of ruin:** {risk_of_ruin*100:.1f}%")
    st.write(f"**Median final bankroll:** {np.median(sims[:, -1]):.2f}")
    st.write(f"**Mean final bankroll:** {sims[:, -1].mean():.2f}")
