from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v5.0
# Name: Home Dashboard
# File: src/app/pages/01_Home.py
# Purpose: Overview of today's card, quick links to core
#          v5 betting tools and status.
# ============================================================

from datetime import date

import streamlit as st

from src.app.ui.header import render_header
from src.app.ui.page_state import set_active_page
from src.app.ui.navbar import render_navbar
from src.app.engines.bet_tracker import load_bets, compute_roi
from src.app.utils.pipeline_status import (
    get_pipeline_last_run,
    get_ingestion_last_run,
)

st.set_page_config(page_title="NBA Analytics v5 – Home", page_icon="🏀", layout="wide")

render_header()
set_active_page("Home")
render_navbar()

st.title("🏠 Home")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Bet Tracker Snapshot")
    df = load_bets()
    stats = compute_roi(df)
    st.metric("Total Bets", stats["total_bets"])
    st.metric("Profit", f"{stats['profit']:.2f}")
    st.metric("ROI", f"{stats['roi']*100:.2f}%")

with col2:
    st.subheader("🩺 Pipeline Status")
    st.write(f"Last Ingestion Run: {get_ingestion_last_run()}")
    st.write(f"Last Pipeline Run: {get_pipeline_last_run()}")

st.markdown("---")

st.subheader("⚙️ Core Tools")
col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown("### 🤖 Automated Recommendations")
    st.write("Model-driven best bets across moneyline, totals, and spread.")
    st.page_link("pages/08_Automated_Recommendations.py", label="Open")

with col_b:
    st.markdown("### 🎲 Parlay Builder")
    st.write("Build, evaluate, and log parlays with EV and win probability.")
    st.page_link("pages/11_Parlay_Builder.py", label="Open")

with col_c:
    st.markdown("### 📒 Bet Tracker")
    st.write("View, filter, and analyze your historical bets.")
    st.page_link("pages/06_Bet_Tracker.py", label="Open")

st.markdown("---")

st.subheader("🔍 Advanced Analytics")
st.write(
    "- 🧪 Simulation Lab: Simulate bankroll paths based on historical bets.\n"
    "- 📈 Backtesting Dashboard: Evaluate model edges historically.\n"
    "- 🧠 Model Registry: Inspect trained models and versions."
)
