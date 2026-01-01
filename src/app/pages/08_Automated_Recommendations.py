from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v5.0
# Name: Automated Recommendations
# File: src/app/pages/08_Automated_Recommendations.py
# Purpose: Central recommendations hub aggregating edges across
#          markets into ranked, actionable bets.
# ============================================================

from datetime import date

import pandas as pd
import streamlit as st

from src.app.ui.header import render_header
from src.app.ui.page_state import set_active_page
from src.app.ui.navbar import render_navbar
from src.app.engines.best_bets import compute_best_bets
from src.app.engines.bet_tracker import BetRecord, append_bet, new_bet_id
from src.app.engines.parlay import ParlayLeg
from src.app.engines.betting_math import calculate_edge

st.set_page_config(
    page_title="Automated Recommendations",
    page_icon="🤖",
    layout="wide",
)

render_header()
set_active_page("Automated Recommendations")
render_navbar()

st.title("🤖 Automated Recommendations")

today = date.today()
pred_date = st.date_input("Prediction date", value=today)

df = compute_best_bets(pred_date)
if df.empty:
    st.info("No recommendations available for this date.")
    st.stop()

st.markdown("### Filters")

col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    min_conf = st.selectbox(
        "Minimum confidence",
        ["Low", "Medium", "High"],
        index=0,
    )
with col_f2:
    min_edge = st.number_input("Minimum edge (decimal)", value=0.02, step=0.01)
with col_f3:
    market_filter = st.multiselect(
        "Markets",
        options=["Moneyline", "Totals", "Spread"],
        default=["Moneyline", "Totals", "Spread"],
    )

conf_rank = {"High": 3, "Medium": 2, "Low": 1, "None": 0}
min_rank = conf_rank[min_conf]

df_view = df[
    (df["confidence_rank"] >= min_rank)
    & (df["edge"] >= min_edge)
    & (df["market"].isin(market_filter))
].copy()

st.markdown("### Ranked Recommendations")

st.dataframe(
    df_view[["market", "team", "opponent", "bet", "odds", "win_prob", "edge", "confidence"]],
    use_container_width=True,
)

st.markdown("---")
st.markdown("### Actions")

if "parlay_legs" not in st.session_state:
    st.session_state["parlay_legs"] = []

for idx, row in df_view.iterrows():
    with st.expander(f"{row['market']} – {row['bet']} ({row['team']} vs {row['opponent']})"):
        st.write(f"Edge: {row['edge']:.3f}")
        if pd.notna(row["win_prob"]):
            st.write(f"Win probability: {row['win_prob']*100:.1f}%")
        if pd.notna(row["odds"]):
            st.write(f"Odds: {row['odds']}")

        col_a, col_b = st.columns(2)

        with col_a:
            can_add = pd.notna(row["odds"]) and pd.notna(row.get("win_prob"))
            if st.button(
                "➕ Add to Parlay",
                key=f"auto_add_parlay_{idx}",
                disabled=not can_add,
            ):
                st.session_state["parlay_legs"].append(
                    ParlayLeg(
                        description=row["bet"],
                        odds=float(row["odds"]),
                        win_prob=float(row["win_prob"]),
                        team=row["team"],
                        opponent=row["opponent"],
                        market=row["market"].lower(),
                        source="automated",
                    )
                )
                st.success("Added to parlay builder.")

        with col_b:
            if st.button(
                "📝 Log Single Bet",
                key=f"auto_log_bet_{idx}",
                disabled=pd.isna(row.get("odds")),
            ):
                record = BetRecord(
                    bet_id=new_bet_id(),
                    date=str(date.today()),
                    game_date=str(pred_date),
                    market=row["market"].lower(),
                    team=row["team"],
                    opponent=row["opponent"],
                    bet_description=row["bet"],
                    odds=float(row["odds"]),
                    stake=100.0,
                    result="pending",
                    payout=0.0,
                    edge=float(row["edge"]),
                    confidence=row["confidence"],
                    confidence_rank=int(row["confidence_rank"]),
                    source="automated",
                )
                append_bet(record)
                st.success(f"Bet logged. Bet ID: {record.bet_id}")
