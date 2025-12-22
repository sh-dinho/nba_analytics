# ============================================================
# 🏀 NBA Analytics v3
# Module: Dashboard — Betting Recommendations
# File: src/dashboard/tabs/recommendations.py
# Author: Sadiq
#
# Description:
#     Displays unified betting recommendations across:
#       - Moneyline
#       - Totals (O/U)
#       - Spread (ATS)
#
#     Powered by:
#       src/markets/recommend.py
# ============================================================

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config.paths import (
    PREDICTIONS_DIR,
    DATA_DIR,
    ODDS_DIR,
)
from src.markets.recommend import generate_recommendations

TOTALS_DIR = DATA_DIR / "predictions_totals"
SPREAD_DIR = DATA_DIR / "predictions_spread"


def _load_ml(pred_date):
    path = PREDICTIONS_DIR / f"predictions_{pred_date}.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _load_totals(pred_date):
    path = TOTALS_DIR / f"totals_{pred_date}.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _load_spread(pred_date):
    path = SPREAD_DIR / f"spread_{pred_date}.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _load_odds(pred_date):
    path = ODDS_DIR / f"odds_{pred_date}.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def render_recommendations_tab():
    st.header("Automated Betting Recommendations")

    pred_date = st.date_input("Prediction date")

    ml = _load_ml(pred_date)
    totals = _load_totals(pred_date)
    spread = _load_spread(pred_date)
    odds = _load_odds(pred_date)

    if ml.empty and totals.empty and spread.empty:
        st.warning("No predictions available for this date.")
        return

    recs = generate_recommendations(
        ml_df=ml,
        totals_df=totals,
        spread_df=spread,
        odds_df=odds,
        start_date=None,
        end_date=None,
    )

    if recs.empty:
        st.info("No recommendations generated.")
        return

    st.subheader("Recommended Bets")
    st.dataframe(
        recs[
            ["market", "team", "recommendation", "edge", "confidence", "risk_flags"]
        ].sort_values("confidence", ascending=False),
        use_container_width=True,
    )

    st.success("Recommendations generated successfully.")
