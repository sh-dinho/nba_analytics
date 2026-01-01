from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v5.0
# Name: Backtesting Dashboard
# File: src/app/pages/07_Backtesting_Dashboard.py
# Purpose: Visualize historical model performance, edges, and
#          ROI across seasons and markets.
# ============================================================

import pandas as pd
import streamlit as st

from src.app.ui.header import render_header
from src.app.ui.page_state import set_active_page
from src.app.ui.navbar import render_navbar
from src.config.paths import DATA_DIR

st.set_page_config(page_title="Backtesting Dashboard", page_icon="📈", layout="wide")

render_header()
set_active_page("Backtesting Dashboard")
render_navbar()

st.title("📈 Backtesting Dashboard")

backtest_path = DATA_DIR / "backtests" / "model_backtest.parquet"

if not backtest_path.exists():
    st.info("No backtest file found. Expected at: backtests/model_backtest.parquet")
    st.stop()

df = pd.read_parquet(backtest_path)

st.markdown("### Filters")

markets = sorted(df["market"].dropna().unique().tolist())
seasons = sorted(df["season"].dropna().unique().tolist())

col_f1, col_f2 = st.columns(2)
with col_f1:
    market = st.selectbox("Market", options=markets)
with col_f2:
    season = st.selectbox("Season", options=["All"] + seasons)

df_view = df[df["market"] == market].copy()
if season != "All":
    df_view = df_view[df_view["season"] == season]

st.markdown("### Backtest Results")
st.dataframe(
    df_view[
        [
            "date",
            "team",
            "opponent",
            "market",
            "odds",
            "win_prob",
            "result",
            "payout",
            "edge",
        ]
    ],
    use_container_width=True,
)

st.markdown("---")
st.markdown("### ROI Over Time")

if "date" in df_view.columns:
    df_view["cum_profit"] = df_view["payout"].cumsum()
    st.line_chart(df_view.set_index("date")["cum_profit"])
else:
    st.write("No date column in backtest file; cannot plot cumulative profit.")
