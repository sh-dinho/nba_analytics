from __future__ import annotations

from datetime import date

import streamlit as st

from src.model.predict import run_prediction_for_date
from src.model.predict_totals import run_totals_prediction_for_date
from src.model.predict_spread import run_spread_prediction_for_date
from src.markets.recommend import implied_prob
from src.app.ui.header import render_header
from src.app.ui.page_state import set_active_page
from src.app.ui.navbar import render_navbar

render_header()
set_active_page("PAGE NAME HERE")
render_navbar()

st.set_page_config(page_title="Bet Recommendations", page_icon="💰", layout="wide")
render_header()
set_active_page("Bet Recommendations")
render_navbar()

st.title("💰 Bet Recommendations Engine")

pred_date = st.date_input("Prediction date", value=date.today())

ml = run_prediction_for_date(pred_date)
tot = run_totals_prediction_for_date(pred_date)
sp = run_spread_prediction_for_date(pred_date)

if ml.empty and tot.empty and sp.empty:
    st.info("No games for this date — no recommendations available.")
    st.stop()

tabs = st.tabs(["🏆 Moneyline", "📊 Totals", "📐 Spread"])

# Moneyline
with tabs[0]:
    st.subheader("Moneyline recommendations")
    if {"moneyline_odds", "win_probability"} <= set(ml.columns):
        ml = ml.copy()
        ml["implied"] = ml["moneyline_odds"].apply(implied_prob)
        ml["edge"] = ml["win_probability"] - ml["implied"]

        min_edge = st.slider("Minimum edge (decimal)", 0.0, 0.2, 0.02, 0.01)
        rec = ml[ml["edge"] >= min_edge].sort_values("edge", ascending=False)

        if rec.empty:
            st.info("No bets meet the edge threshold.")
        else:
            st.dataframe(
                rec[
                    [
                        "game_id",
                        "team",
                        "opponent",
                        "win_probability",
                        "moneyline_odds",
                        "edge",
                    ]
                ],
                use_container_width=True,
            )
    else:
        st.info("Sportsbook moneyline odds not available.")

# Totals
with tabs[1]:
    st.subheader("Totals recommendations")
    if {"total_line", "predicted_total_points"} <= set(tot.columns):
        tot = tot.copy()
        tot["diff"] = tot["predicted_total_points"] - tot["total_line"]

        min_diff = st.slider("Minimum absolute diff (points)", 0.0, 10.0, 2.0, 0.5)
        rec = tot[tot["diff"].abs() >= min_diff].sort_values("diff", ascending=False)

        if rec.empty:
            st.info("No totals edges meet the threshold.")
        else:
            st.dataframe(
                rec[
                    [
                        "game_id",
                        "home_team",
                        "away_team",
                        "predicted_total_points",
                        "total_line",
                        "diff",
                    ]
                ],
                use_container_width=True,
            )
    else:
        st.info("Sportsbook totals not available.")

# Spread
with tabs[2]:
    st.subheader("Spread recommendations")
    if {"spread_line", "predicted_margin"} <= set(sp.columns):
        sp = sp.copy()
        sp["edge"] = sp["predicted_margin"] - sp["spread_line"]

        min_edge_pts = st.slider("Minimum absolute edge (points)", 0.0, 10.0, 2.0, 0.5)
        rec = sp[sp["edge"].abs() >= min_edge_pts].sort_values("edge", ascending=False)

        if rec.empty:
            st.info("No spread edges meet the threshold.")
        else:
            st.dataframe(
                rec[
                    [
                        "game_id",
                        "home_team",
                        "away_team",
                        "predicted_margin",
                        "spread_line",
                        "edge",
                    ]
                ],
                use_container_width=True,
            )
    else:
        st.info("Sportsbook spreads not available.")

st.markdown("---")
st.info(
    "For stability‑adjusted ML recommendations, open the **Automated Recommendations** page."
)
