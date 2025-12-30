from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v4
# Page: Automated Recommendations
# File: src/app/pages/11_Automated_Recommendations.py
#
# Description:
#     Interactive dashboard for the v4 automated recommendations:
#       - runs v4 predictors + loads odds
#       - calls src.markets.recommend.generate_recommendations
#       - allows filtering by edge / stability / confidence
#       - supports:
#           * Add to Parlay (session_state["parlay_legs"])
#           * Log Bet to Bet Tracker
# ============================================================

from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config.paths import ODDS_DIR
from src.markets.recommend import (
    generate_recommendations,
    summarize_recommendations_for_dashboard,
    RecommendationConfig,
)
from src.model.predict import run_prediction_for_date
from src.model.predict_totals import run_totals_prediction_for_date
from src.model.predict_spread import run_spread_prediction_for_date
from src.app.engines.parlay import ParlayLeg
from src.app.engines.bet_tracker import BetRecord, append_bet, new_bet_id


from src.app.ui.header import render_header
from src.app.ui.page_state import set_active_page
from src.app.ui.navbar import render_navbar

render_header()
set_active_page("PAGE NAME HERE")
render_navbar()


st.set_page_config(
    page_title="Automated Recommendations", page_icon="🤖", layout="wide"
)
render_header()

st.title("🤖 Automated Betting Recommendations (v4)")

# ------------------------------------------------------------
# Odds loader (simple convention-based; adjust if needed)
# ------------------------------------------------------------


def load_odds_for_date(pred_date: date) -> pd.DataFrame:
    """
    Load odds for a given date.
    Expects files like: data/odds/odds_YYYY-MM-DD.parquet or .csv.
    Returns empty dataframe if not found.
    """
    base = f"odds_{pred_date.isoformat()}"
    parquet_path = ODDS_DIR / f"{base}.parquet"
    csv_path = ODDS_DIR / f"{base}.csv"

    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)

    return pd.DataFrame(columns=["game_id", "price"])


# ------------------------------------------------------------
# Date + config controls
# ------------------------------------------------------------

col_date, col_prob, col_edge, col_stab = st.columns(4)

with col_date:
    pred_date = st.date_input("Prediction date", value=date.today())
with col_prob:
    min_prob = st.slider("Min win probability", 0.5, 0.8, 0.55, 0.01)
with col_edge:
    min_edge = st.slider("Min ML edge (decimal)", 0.0, 0.2, 0.02, 0.01)
with col_stab:
    min_stability = st.slider("Min stability score", 0.0, 1.0, 0.0, 0.05)

cfg = RecommendationConfig(
    min_probability=min_prob,
    min_edge=min_edge,
    min_stability=min_stability,
)

if "parlay_legs" not in st.session_state:
    st.session_state["parlay_legs"] = []

# ------------------------------------------------------------
# Load predictions + odds, run engine
# ------------------------------------------------------------

ml = run_prediction_for_date(pred_date)
totals = run_totals_prediction_for_date(pred_date)
spread = run_spread_prediction_for_date(pred_date)
odds = load_odds_for_date(pred_date)

if ml.empty:
    st.info("No moneyline predictions for this date — cannot generate recommendations.")
    st.stop()

recs = generate_recommendations(ml, totals, spread, odds, cfg=cfg)

if recs.empty:
    st.warning("No recommendations after applying filters.")
    st.stop()

summary = summarize_recommendations_for_dashboard(recs)

st.markdown("### Summary table")


def color_confidence(val: int) -> str:
    if val >= 80:
        return "background-color: #2ecc71; color: white;"
    if val >= 60:
        return "background-color: #f1c40f; color: black;"
    if val >= 40:
        return "background-color: #e67e22; color: white;"
    return ""


display_cols = [
    "matchup",
    "team",
    "bet",
    "confidence_index",
    "edge_pct",
    "win_probability",
    "stability_score",
    "price",
]
styled = summary[display_cols].style.applymap(
    color_confidence, subset=["confidence_index"]
)
st.dataframe(styled, use_container_width=True)

st.markdown("---")
st.markdown("### Actions per recommendation")

stake_default = st.number_input(
    "Default stake for 'Log Bet' (can be overridden manually elsewhere)",
    value=100.0,
    step=10.0,
)

for _, row in summary.iterrows():
    game_id = row["game_id"]
    team = row["team"]
    matchup = row["matchup"]
    bet_text = row["bet"]
    confidence = row["confidence_index"]
    price = row["price"]
    win_prob_pct = row["win_probability"]  # already in %
    win_prob = float(win_prob_pct) / 100.0

    with st.expander(f"{matchup} — {team} ({confidence} conf)"):
        st.write(bet_text)
        st.write(f"Edge: {row['edge_pct']:.1f}%")
        st.write(f"Win probability: {row['win_probability']:.1f}%")
        st.write(f"Stability score: {row['stability_score']:.2f}")
        st.write(f"Price (American): {price if pd.notna(price) else 'n/a'}")

        col1, col2 = st.columns(2)

        # Add to Parlay
        with col1:
            can_add_parlay = pd.notna(price)
            if st.button(
                "➕ Add to Parlay",
                key=f"add_parlay_{game_id}_{team}",
                disabled=not can_add_parlay,
            ):
                leg = ParlayLeg(
                    description=f"{team} ML ({matchup})",
                    odds=float(price),
                    win_prob=win_prob,
                )
                st.session_state["parlay_legs"].append(leg)
                st.success("Added to parlay.")

        # Log Bet directly
        with col2:
            can_log = pd.notna(price)
            if st.button(
                "📝 Log Bet",
                key=f"log_bet_{game_id}_{team}",
                disabled=not can_log,
            ):
                record = BetRecord(
                    bet_id=new_bet_id(),
                    date=str(date.today()),
                    game_date=str(pred_date),
                    market="moneyline",
                    team=team,
                    opponent=matchup.replace(f"{team} vs ", ""),
                    bet_description=f"{team} ML (auto rec)",
                    odds=float(price),
                    stake=float(stake_default),
                    result="pending",
                    payout=0.0,
                    edge=float(row["edge_pct"] / 100.0),
                    confidence=(
                        "High"
                        if confidence >= 80
                        else "Medium" if confidence >= 60 else "Low"
                    ),
                )
                append_bet(record)
                st.success(f"Bet logged to Bet Tracker. Bet ID: {record.bet_id}")

st.markdown("---")
st.info(
    "Legs you add here are stored in session. "
    "Open the **Parlay Builder** page to see them, compute EV, apply Kelly sizing, "
    "and log the parlay into your Bet Tracker."
)
