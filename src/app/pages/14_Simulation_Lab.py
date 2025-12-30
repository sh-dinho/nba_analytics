from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v4
# Page: Simulation Lab
# File: src/app/pages/14_Simulation_Lab.py
#
# Description:
#     Monte Carlo simulation lab for bankroll and performance:
#       - uses historical bet outcomes
#       - simulates future paths
#       - supports flat and Kelly-based staking
# ============================================================

from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

from src.app.ui.header import render_header
from src.app.ui.page_state import set_active_page
from src.app.ui.navbar import render_navbar
from src.app.engines.bet_tracker import load_bets, simulate_bankroll, kelly_fraction

from src.app.ui.header import render_header
from src.app.ui.page_state import set_active_page
from src.app.ui.navbar import render_navbar

render_header()
set_active_page("PAGE NAME HERE")
render_navbar()


st.markdown(
    "Experiment with bankroll behavior using your historical results. "
    "Simulate future paths under different staking strategies."
)

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

strategy = st.selectbox("Staking strategy", ["Flat stake", "Kelly fraction (approx)"])

flat_stake = (
    st.number_input("Flat stake per bet", value=50.0, min_value=0.0)
    if strategy == "Flat stake"
    else None
)
kelly_frac_override = (
    st.number_input(
        "Global Kelly fraction (0-1)", value=0.03, min_value=0.0, max_value=1.0
    )
    if strategy == "Kelly fraction (approx)"
    else None
)

st.markdown("---")
st.markdown("### Run simulation")

if st.button("Run simulation"):
    payouts = resolved["payout"].to_numpy(dtype=float)
    stakes = resolved["stake"].to_numpy(dtype=float)
    odds = resolved["odds"].to_numpy(dtype=float)

    if strategy == "Flat stake":
        # Rescale historical payouts to match chosen flat stake
        # Assume proportional scaling: payout ~ stake
        median_stake = np.median(stakes) if len(stakes) > 0 else 1.0
        scale = flat_stake / median_stake if median_stake > 0 else 1.0
        payouts_scaled = payouts * scale

        sims = simulate_bankroll(
            payouts=payouts_scaled,
            initial_bankroll=initial_bankroll,
            horizon=int(horizon),
            n_sims=int(n_sims),
        )

    else:
        # Kelly fraction strategy:
        # approximate per-bet Kelly fraction from historical implied win_prob and odds,
        # then scale by a chosen global factor (kelly_frac_override)
        # For simplicity, treat each bet as same Kelly fraction (risk-controlled).
        mean_kelly = 0.03
        if len(resolved) > 0:
            # rough proxy: calibrate to Sharpe-like behavior using payout/stake
            returns = resolved["payout"] / resolved["stake"]
            mu = returns.mean()
            sigma = returns.std()
            if sigma > 0:
                mean_kelly = max(mu / (sigma**2), 0.0)
                mean_kelly = min(mean_kelly, 0.25)

        effective_kelly = mean_kelly * (
            kelly_frac_override if kelly_frac_override is not None else 1.0
        )

        sims = np.zeros((int(n_sims), int(horizon)), dtype=float)
        for s in range(int(n_sims)):
            bankroll = initial_bankroll
            for t in range(int(horizon)):
                idx = np.random.randint(0, len(payouts))
                # scaled stake as fraction of current bankroll
                stake_t = bankroll * effective_kelly
                # scale payout relative to original stake
                original_stake = stakes[idx] if stakes[idx] > 0 else 1.0
                payout_scale = stake_t / original_stake
                bankroll += payouts[idx] * payout_scale
                sims[s, t] = bankroll

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
