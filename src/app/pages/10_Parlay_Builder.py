from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v4
# Page: Parlay Builder
# File: src/app/pages/10_Parlay_Builder.py
#
# Description:
#     Builds parlays from:
#       - legs pushed from Best Bets (session_state)
#       - manual leg definitions
#     Computes:
#       - combined win probability
#       - decimal odds
#       - EV
#       - Kelly sizing
#     Can log parlays into the Bet Tracker.
# ============================================================

import streamlit as st

from src.app.ui.header import render_header
from src.app.engines.parlay import ParlayLeg, parlay_expected_value
from src.app.engines.parlay_to_bettracker import log_parlay

from src.app.ui.header import render_header
from src.app.ui.page_state import set_active_page
from src.app.ui.navbar import render_navbar

render_header()
set_active_page("PAGE NAME HERE")
render_navbar()


st.title("🎲 Parlay Builder")

st.markdown(
    "Use legs added from **Best Bets** or define new ones here. "
    "Then compute EV, Kelly sizing, and log the parlay into your Bet Tracker."
)

if "parlay_legs" not in st.session_state:
    st.session_state["parlay_legs"] = []

existing_legs = st.session_state["parlay_legs"]

st.markdown("### Current parlay legs (from session)")

if existing_legs:
    for i, leg in enumerate(existing_legs):
        st.markdown(
            f"- **{i+1}.** {leg.description} — odds: {leg.odds:.0f}, "
            f"win prob: {leg.win_prob*100:.1f}%"
        )
else:
    st.info(
        "No legs currently in parlay. Add legs from **Best Bets** or define them below."
    )

st.markdown("---")

st.markdown("### Define / edit parlay legs")

num_legs = st.slider(
    "Number of legs (including preloaded ones)",
    min_value=1,
    max_value=10,
    value=max(1, len(existing_legs)),
)
stake = st.number_input("Stake", min_value=0.0, step=1.0, value=100.0)

legs: list[ParlayLeg] = []

for i in range(num_legs):
    st.markdown(f"#### Leg {i+1}")
    col1, col2, col3 = st.columns(3)
    pre = existing_legs[i] if i < len(existing_legs) else None

    with col1:
        desc = st.text_input(
            f"Description {i+1}",
            key=f"desc_{i}",
            value=pre.description if pre else "",
        )
    with col2:
        odds = st.number_input(
            f"Odds {i+1} (American)",
            key=f"odds_{i}",
            value=float(pre.odds) if pre else -110.0,
        )
    with col3:
        win_prob = st.number_input(
            f"Model win prob {i+1} (0-1)",
            key=f"prob_{i}",
            min_value=0.0,
            max_value=1.0,
            value=float(pre.win_prob) if pre else 0.55,
        )

    legs.append(ParlayLeg(description=desc, odds=float(odds), win_prob=float(win_prob)))

if st.button("Compute parlay EV"):
    result = parlay_expected_value(legs, stake)
    st.markdown("### Parlay summary")
    st.write("Combined win probability:", f"{result['win_prob']*100:.1f}%")
    st.write("Decimal odds:", f"{result['decimal_odds']:.3f}")
    st.write("Expected value (EV):", f"{result['ev']:.2f}")

    # Kelly sizing for parlay
    st.markdown("### 🎯 Kelly sizing (parlay)")

    colA, colB = st.columns(2)
    with colA:
        k_bankroll = st.number_input(
            "Bankroll", value=1000.0, min_value=0.0, key="parlay_kelly_bankroll"
        )
    with colB:
        k_win_prob = result["win_prob"]

    b = result["decimal_odds"] - 1.0
    q = 1 - k_win_prob
    kelly_f = (b * k_win_prob - q) / b if b > 0 else 0.0
    kelly_f = max(float(kelly_f), 0.0)

    st.write(f"**Kelly fraction:** {kelly_f:.3f}")
    st.write(f"**Recommended stake:** {kelly_f * k_bankroll:.2f}")

    if st.button("Log parlay to Bet Tracker"):
        record = log_parlay(
            legs=legs,
            stake=stake,
            win_prob=result["win_prob"],
            decimal_odds=result["decimal_odds"],
            ev=result["ev"],
        )
        st.success(f"Parlay logged to Bet Tracker. Bet ID: {record.bet_id}")

st.markdown("---")
st.caption(
    "Parlays are volatile. Use Kelly sizing and bankroll simulation to manage risk."
)
