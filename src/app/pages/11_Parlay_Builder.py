from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v5.0
# Name: Parlay Builder
# File: src/app/pages/11_Parlay_Builder.py
# Purpose: Interactive parlay construction, EV and win-prob
#          calculations, and logging to Bet Tracker.
# ============================================================

from datetime import date

import streamlit as st

from src.app.ui.header import render_header
from src.app.ui.page_state import set_active_page
from src.app.ui.navbar import render_navbar
from src.app.engines.parlay import ParlayLeg, parlay_expected_value
from src.app.engines.parlay_to_bettracker import log_parlay

st.set_page_config(page_title="Parlay Builder", page_icon="🎲", layout="wide")

render_header()
set_active_page("Parlay Builder")
render_navbar()

st.title("🎲 Parlay Builder")

if "parlay_legs" not in st.session_state:
    st.session_state["parlay_legs"] = []

legs: list[ParlayLeg] = st.session_state["parlay_legs"]

st.markdown("### Current Legs")

if not legs:
    st.info("No legs in your parlay. Add legs from other pages or manually.")
else:
    for i, leg in enumerate(legs):
        st.write(
            f"{i+1}. {leg.description} @ {leg.odds} "
            f"(p={leg.win_prob*100:.1f}%)"
        )

    remove_idx = st.number_input(
        "Remove leg index (1-based, optional)",
        min_value=1,
        max_value=len(legs),
        value=1,
    )
    if st.button("Remove Selected Leg"):
        legs.pop(remove_idx - 1)
        st.session_state["parlay_legs"] = legs
        st.experimental_rerun()

st.markdown("---")
st.markdown("### Add Manual Leg")

desc = st.text_input("Leg Description")
odds = st.number_input("Odds (American)", value=-110.0)
win_prob = st.number_input(
    "Win Probability (0–1)", value=0.55, min_value=0.0, max_value=1.0
)

if st.button("Add Manual Leg"):
    legs.append(
        ParlayLeg(
            description=desc,
            odds=float(odds),
            win_prob=float(win_prob),
            source="manual_parlay_builder",
        )
    )
    st.session_state["parlay_legs"] = legs
    st.success("Manual leg added.")
    st.experimental_rerun()

st.markdown("---")
st.markdown("### Parlay Metrics")

stake = st.number_input("Stake", value=50.0, min_value=0.0)

if not legs:
    st.info("Add at least one leg to compute EV.")
else:
    stats = parlay_expected_value(legs, stake=stake)
    st.write(f"**Win probability:** {stats['win_prob']*100:.1f}%")
    st.write(f"**Decimal odds:** {stats['decimal_odds']:.2f}")
    st.write(f"**American odds:** {stats['american_odds']:.0f}")
    st.write(f"**Expected value:** {stats['ev']:.2f}")

    if st.button("📝 Log Parlay as Bet"):
        record = log_parlay(
            legs=legs,
            stake=stake,
            win_prob=stats["win_prob"],
            decimal_odds=stats["decimal_odds"],
            ev=stats["ev"],
            source="parlay_builder",
        )
        st.success(f"Parlay logged. Bet ID: {record.bet_id}")
