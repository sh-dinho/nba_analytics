from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v5.0
# Name: Smart Parlay Generator
# File: src/app/pages/10_Smart_Parlay_Generator.py
# Purpose: Generate candidate parlays from recommendations,
#          avoiding correlations and maximizing EV.
# ============================================================

from datetime import date

import pandas as pd
import streamlit as st

from src.app.ui.header import render_header
from src.app.ui.page_state import set_active_page
from src.app.ui.navbar import render_navbar
from src.app.engines.best_bets import compute_best_bets
from src.app.engines.parlay import ParlayLeg, parlay_expected_value

st.set_page_config(
    page_title="Smart Parlay Generator",
    page_icon="🎯",
    layout="wide",
)

render_header()
set_active_page("Smart Parlay Generator")
render_navbar()

st.title("🎯 Smart Parlay Generator")

today = date.today()
pred_date = st.date_input("Prediction date", value=today)
stake = st.number_input("Parlay stake", value=50.0, min_value=0.0)

df = compute_best_bets(pred_date)
if df.empty:
    st.info("No recommendations available.")
    st.stop()

st.markdown("### Parlay Settings")
max_legs = st.slider("Maximum legs", min_value=2, max_value=8, value=4)
min_edge = st.number_input("Minimum edge per leg", value=0.03, step=0.01)

df = df[
    (df["edge"] >= min_edge) & df["market"].isin(["Moneyline", "Totals", "Spread"])
].copy()

if df.empty:
    st.info("No legs meet the criteria.")
    st.stop()

st.markdown("### Candidate Legs")
st.dataframe(
    df[["market", "team", "opponent", "bet", "odds", "win_prob", "edge"]],
    use_container_width=True,
)

# Simple heuristic: sort by edge and take top N non-correlated (different games / teams)
# Assuming team+opponent as proxy for game identity here:
df["game_key"] = df["team"] + "_" + df["opponent"]
used_games = set()
legs: list[ParlayLeg] = []

for _, r in df.sort_values("edge", ascending=False).iterrows():
    if len(legs) >= max_legs:
        break
    if r["game_key"] in used_games:
        continue
    if pd.isna(r["odds"]) or pd.isna(r["win_prob"]):
        continue
    used_games.add(r["game_key"])
    legs.append(
        ParlayLeg(
            description=r["bet"],
            odds=float(r["odds"]),
            win_prob=float(r["win_prob"]),
            team=r["team"],
            opponent=r["opponent"],
            market=r["market"].lower(),
            source="smart_parlay",
        )
    )

if not legs:
    st.info("No valid parlay legs could be constructed.")
    st.stop()

parlay_stats = parlay_expected_value(legs, stake=stake)

st.markdown("### Suggested Parlay")
for leg in legs:
    st.write(f"- {leg.description} @ {leg.odds} (p={leg.win_prob*100:.1f}%)")

st.write(f"**Combined win probability:** {parlay_stats['win_prob']*100:.1f}%")
st.write(f"**Combined odds (decimal):** {parlay_stats['decimal_odds']:.2f}")
st.write(f"**Combined odds (American):** {parlay_stats['american_odds']:.0f}")
st.write(f"**Expected value:** {parlay_stats['ev']:.2f}")

if "parlay_legs" not in st.session_state:
    st.session_state["parlay_legs"] = []

if st.button("➕ Send to Parlay Builder"):
    st.session_state["parlay_legs"] = legs
    st.success("Parlay legs sent to Parlay Builder.")
