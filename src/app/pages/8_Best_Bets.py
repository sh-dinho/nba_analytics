from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st


from src.app.engines.parlay import ParlayLeg
from src.app.engines.best_bets import compute_best_bets  # if you keep this helper
from src.app.ui.header import render_header
from src.app.ui.page_state import set_active_page
from src.app.ui.navbar import render_navbar

render_header()
set_active_page("PAGE NAME HERE")
render_navbar()

st.set_page_config(page_title="Best Bets", page_icon="🔥", layout="wide")
render_header()
set_active_page("Best Bets")
render_navbar()

st.title("🔥 Best Bets of the Day")

pred_date = st.date_input("Prediction date", value=date.today())

if "parlay_legs" not in st.session_state:
    st.session_state["parlay_legs"] = []

bets = compute_best_bets(pred_date)

if bets.empty:
    st.info("No positive-edge best bets for this date.")
    st.stop()


def color_confidence(val: str) -> str:
    if val == "High":
        return "background-color: #2ecc71; color: white;"
    if val == "Medium":
        return "background-color: #f1c40f; color: black;"
    if val == "Low":
        return "background-color: #e67e22; color: white;"
    return ""


st.markdown("### Best bets table")

display_cols = ["market", "team", "opponent", "bet", "edge", "confidence"]
extra_cols = [c for c in ["odds", "win_prob"] if c in bets.columns]
display_cols += extra_cols

styled = bets[display_cols].style.applymap(color_confidence, subset=["confidence"])
st.dataframe(styled, use_container_width=True)

st.markdown("### Add legs to your parlay")

for idx, row in bets.iterrows():
    col1, col2, col3, col4 = st.columns([3, 3, 2, 2])
    with col1:
        st.markdown(f"**{row['market']}** — {row['team']} vs {row['opponent']}")
        st.markdown(f"*Bet:* {row['bet']}")
    with col2:
        st.markdown(f"*Edge:* {row['edge']:.3f}")
        st.markdown(f"*Confidence:* {row['confidence']}")
    with col3:
        if "odds" in bets.columns and not pd.isna(row["odds"]):
            st.markdown(f"*Odds:* {row['odds']:.0f}")
        else:
            st.markdown("*Odds:* n/a")
        if "win_prob" in bets.columns and not pd.isna(row["win_prob"]):
            st.markdown(f"*Win prob:* {row['win_prob']*100:.1f}%")
        else:
            st.markdown("*Win prob:* n/a")
    with col4:
        can_add = (
            ("odds" in bets.columns)
            and ("win_prob" in bets.columns)
            and not pd.isna(row.get("odds"))
            and not pd.isna(row.get("win_prob"))
        )
        if st.button("➕ Add to Parlay", key=f"add_parlay_{idx}", disabled=not can_add):
            leg = ParlayLeg(
                description=f"{row['bet']} ({row['team']} vs {row['opponent']})",
                odds=float(row["odds"]),
                win_prob=float(row["win_prob"]),
            )
            st.session_state["parlay_legs"].append(leg)
            st.success("Added to parlay.")

st.markdown("---")

if st.session_state["parlay_legs"]:
    st.markdown("#### Current parlay legs (session)")
    summary = [
        {
            "description": leg.description,
            "odds": leg.odds,
            "win_prob": leg.win_prob,
        }
        for leg in st.session_state["parlay_legs"]
    ]
    st.dataframe(pd.DataFrame(summary), use_container_width=True)

st.markdown("---")
st.info(
    "Use **Parlay Builder** to turn these legs into an EV‑evaluated parlay and log it to the Bet Tracker."
)
