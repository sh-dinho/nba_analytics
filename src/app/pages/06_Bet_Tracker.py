from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v5.0
# Name: Bet Tracker
# File: src/app/pages/06_Bet_Tracker.py
# Purpose: Display, filter, and analyze logged bets with ROI
#          and Sharpe metrics.
# ============================================================

from datetime import datetime

import pandas as pd
import streamlit as st

from src.app.ui.header import render_header
from src.app.ui.page_state import set_active_page
from src.app.ui.navbar import render_navbar
from src.app.engines.bet_tracker import (
    load_bets,
    compute_roi,
    compute_sharpe_ratio,
    update_bet_result,
)

st.set_page_config(page_title="Bet Tracker", page_icon="📒", layout="wide")

render_header()
set_active_page("Bet Tracker")
render_navbar()

st.title("📒 Bet Tracker")

df = load_bets()
if df.empty:
    st.info("No bets logged yet.")
    st.stop()

st.markdown("### Filters")

col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    market_filter = st.multiselect(
        "Markets",
        options=sorted(df["market"].dropna().unique().tolist()),
        default=sorted(df["market"].dropna().unique().tolist()),
    )
with col_f2:
    result_filter = st.multiselect(
        "Results",
        options=["pending", "win", "loss", "push"],
        default=["pending", "win", "loss", "push"],
    )
with col_f3:
    min_date_str = st.text_input("Min Date (YYYY-MM-DD)", value="")

df_view = df.copy()
if market_filter:
    df_view = df_view[df_view["market"].isin(market_filter)]
if result_filter:
    df_view = df_view[df_view["result"].isin(result_filter)]
if min_date_str:
    try:
        min_date = datetime.strptime(min_date_str, "%Y-%m-%d").date()
        df_view = df_view[df_view["date"] >= str(min_date)]
    except ValueError:
        st.warning("Invalid date format; expected YYYY-MM-DD.")

st.markdown("### Bets")
st.dataframe(df_view, use_container_width=True)

st.markdown("---")
st.markdown("### Summary Stats")

stats = compute_roi(df_view)
sharpe = compute_sharpe_ratio(df_view)

col_s1, col_s2, col_s3 = st.columns(3)
with col_s1:
    st.metric("Total Bets", stats["total_bets"])
    st.metric("Won", stats["won"])
    st.metric("Lost", stats["lost"])
with col_s2:
    st.metric("Staked", f"{stats['staked']:.2f}")
    st.metric("Profit", f"{stats['profit']:.2f}")
with col_s3:
    st.metric("ROI", f"{stats['roi']*100:.2f}%")
    st.metric("Sharpe (per bet)", f"{sharpe:.2f}")

st.markdown("---")
st.markdown("### Update Bet Result")

bet_ids = df[df["result"] == "pending"]["bet_id"].tolist()
if not bet_ids:
    st.write("No pending bets.")
else:
    col_u1, col_u2, col_u3 = st.columns(3)
    with col_u1:
        bet_id = st.selectbox("Bet ID", bet_ids)
    with col_u2:
        new_result = st.selectbox("Result", ["win", "loss", "push"])
    with col_u3:
        if st.button("Update Result"):
            update_bet_result(bet_id, new_result)
            st.success(f"Updated bet {bet_id} to {new_result}.")
            st.experimental_rerun()
