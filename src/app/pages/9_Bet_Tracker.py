from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v4
# Page: Bet Tracker & ROI
# File: src/app/pages/9_Bet_Tracker.py
#
# Description:
#     Full bet lifecycle analytics:
#       - log new bets
#       - update results
#       - ROI metrics
#       - Sharpe ratio
#       - Kelly sizing helper
#       - cumulative profit chart
#       - bankroll Monte Carlo simulator
# ============================================================

from datetime import date

import numpy as np
import streamlit as st

from src.app.ui.header import render_header
from src.app.engines.bet_tracker import (
    BetRecord,
    load_bets,
    append_bet,
    update_bet_result,
    compute_roi,
    compute_sharpe_ratio,
    kelly_fraction,
    simulate_bankroll,
    new_bet_id,
)
from src.app.ui.header import render_header
from src.app.ui.page_state import set_active_page
from src.app.ui.navbar import render_navbar

render_header()
set_active_page("PAGE NAME HERE")
render_navbar()

st.set_page_config(page_title="Bet Tracker", page_icon="📒", layout="wide")
render_header()

st.title("📒 Bet Tracker & ROI")

# ------------------------------------------------------------
# Log a new bet
# ------------------------------------------------------------
st.subheader("Log a new bet")

with st.form("new_bet_form"):
    col1, col2 = st.columns(2)
    with col1:
        game_date = st.date_input("Game date", value=date.today())
        market = st.selectbox("Market", ["moneyline", "totals", "spread", "parlay"])
        team = st.text_input("Team (or MULTI for parlay)", value="")
        opponent = st.text_input("Opponent (or MULTI for parlay)", value="")
    with col2:
        bet_description = st.text_input("Bet description (e.g. LAL ML, Over 221.5)")
        odds = st.number_input("Odds (American)", step=1.0, value=-110.0)
        stake = st.number_input("Stake", min_value=0.0, step=1.0, value=100.0)
        edge = st.number_input(
            "Model edge (optional, decimal, e.g. 0.05 = 5%)", step=0.01, value=0.0
        )
        confidence = st.selectbox("Confidence", ["", "High", "Medium", "Low"])

    submitted = st.form_submit_button("Add bet")

    if submitted:
        record = BetRecord(
            bet_id=new_bet_id(),
            date=str(date.today()),
            game_date=str(game_date),
            market=market,
            team=team,
            opponent=opponent,
            bet_description=bet_description,
            odds=float(odds),
            stake=float(stake),
            result="pending",
            payout=0.0,
            edge=float(edge) if edge else None,
            confidence=confidence or None,
        )
        append_bet(record)
        st.success(f"Bet logged. Bet ID: {record.bet_id}")

st.markdown("---")

# ------------------------------------------------------------
# Bets table + update results
# ------------------------------------------------------------
st.subheader("Your bets")

df = load_bets()
if df.empty:
    st.info("No bets logged yet.")
    st.stop()

st.dataframe(df, use_container_width=True)

st.markdown("### Update bet result")

bet_ids = list(df["bet_id"].values)
col1, col2, col3 = st.columns(3)
with col1:
    sel_bet_id = st.selectbox("Bet ID", bet_ids)
with col2:
    result = st.selectbox("Result", ["win", "loss", "push"])
with col3:
    if st.button("Update result"):
        update_bet_result(bet_id=sel_bet_id, result=result)
        st.success("Bet updated. Refresh page to see changes.")

st.markdown("---")

# ------------------------------------------------------------
# ROI summary + Sharpe
# ------------------------------------------------------------
st.subheader("Performance metrics")

metrics = compute_roi(df)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total bets", metrics["total_bets"])
c2.metric(
    "Won / Lost / Push",
    f"{metrics['won']} / {metrics['lost']} / {metrics['push']}",
)
c3.metric("Total staked", f"{metrics['staked']:.2f}")
c4.metric("Profit / ROI", f"{metrics['profit']:.2f} ({metrics['roi']*100:.1f}%)")

st.markdown("### 📐 Sharpe ratio")

sharpe = compute_sharpe_ratio(df)
st.metric("Sharpe", f"{sharpe:.3f}")
if sharpe > 1:
    st.success("Strong risk-adjusted performance.")
elif sharpe > 0.3:
    st.info("Moderate performance.")
else:
    st.warning("Low risk-adjusted performance.")

# ------------------------------------------------------------
# Kelly sizing helper (single bet)
# ------------------------------------------------------------
st.markdown("### 🎯 Kelly sizing (single bet)")

colA, colB, colC = st.columns(3)
with colA:
    k_odds = st.number_input("Odds (American)", value=-110.0, key="kelly_odds")
with colB:
    k_prob = st.number_input(
        "Win probability (0-1)",
        value=0.55,
        min_value=0.0,
        max_value=1.0,
        key="kelly_prob",
    )
with colC:
    k_bankroll = st.number_input(
        "Bankroll", value=1000.0, min_value=0.0, key="kelly_bankroll"
    )

kelly_f = kelly_fraction(k_prob, k_odds)
kelly_bet = kelly_f * k_bankroll

st.write(f"**Kelly fraction:** {kelly_f:.3f}")
st.write(f"**Recommended stake:** {kelly_bet:.2f}")

st.markdown("---")

# ------------------------------------------------------------
# Cumulative profit chart
# ------------------------------------------------------------
st.subheader("📈 Cumulative profit over time")

resolved = df[df["result"].isin(["win", "loss", "push"])].copy()
if resolved.empty:
    st.info("No resolved bets yet for cumulative chart.")
else:
    resolved = resolved.sort_values("date")
    resolved["cumulative_profit"] = resolved["payout"].cumsum()
    st.line_chart(resolved[["cumulative_profit"]])

# ------------------------------------------------------------
# Bankroll simulator
# ------------------------------------------------------------
st.subheader("🎲 Bankroll simulator (Monte Carlo)")

resolved_sim = df[df["result"].isin(["win", "loss"])].copy()
if len(resolved_sim) < 5:
    st.info("Need at least 5 resolved bets to simulate bankroll.")
else:
    colA, colB = st.columns(2)
    with colA:
        sim_bankroll = st.number_input(
            "Initial bankroll", value=1000.0, min_value=0.0, key="sim_bankroll"
        )
    with colB:
        sim_horizon = st.number_input(
            "Simulation horizon (bets)",
            value=200,
            min_value=10,
            max_value=1000,
            key="sim_horizon",
        )

    payouts = resolved_sim["payout"].to_numpy(dtype=float)
    sims = simulate_bankroll(
        payouts=payouts,
        initial_bankroll=sim_bankroll,
        horizon=int(sim_horizon),
        n_sims=3000,
    )

    mean_path = sims.mean(axis=0)
    p10 = np.percentile(sims, 10, axis=0)
    p90 = np.percentile(sims, 90, axis=0)

    st.line_chart(
        {
            "Mean": mean_path,
            "10th percentile": p10,
            "90th percentile": p90,
        }
    )

    risk_of_ruin = float((sims[:, -1] <= 0).mean())
    st.write(f"**Risk of ruin:** {risk_of_ruin*100:.1f}%")

st.markdown("---")
st.caption("Logged bets + ROI turn your model edges into real performance analytics.")
