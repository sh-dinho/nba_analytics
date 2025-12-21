# ============================================================
# Project: NBA Analytics & Betting Engine
# Module: Dashboard App
# Author: Sadiq
#
# Description:
#     Streamlit dashboard for inspecting:
#       - Daily predictions
#       - Odds + predictions merged view
#       - Historical backtest performance (bankroll, ROI, drawdown)
#
# ============================================================

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.config.paths import PREDICTIONS_DIR, ODDS_DIR
from src.backtest.engine import Backtester, BacktestConfig


# ------------------------------------------------------------
# Data loading helpers
# ------------------------------------------------------------


def _load_predictions(pred_date: date) -> pd.DataFrame:
    path = PREDICTIONS_DIR / f"predictions_{pred_date}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _load_odds(pred_date: date) -> pd.DataFrame:
    path = ODDS_DIR / f"odds_{pred_date}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


# ------------------------------------------------------------
# Plot helpers
# ------------------------------------------------------------


def plot_bankroll_curve(records: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4))
    records = records.sort_values("date")
    ax.plot(records["date"], records["bankroll_after"], marker="o")
    ax.set_title("Bankroll Over Time")
    ax.set_ylabel("Bankroll")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------
# Tabs
# ------------------------------------------------------------


def predictions_tab():
    st.header("Predictions")

    pred_date = st.date_input("Date", value=date.today(), key="pred_date")

    preds = _load_predictions(pred_date)
    odds = _load_odds(pred_date)

    if preds.empty:
        st.warning(f"No predictions found for {pred_date}.")
        return

    st.subheader("Model Predictions (team-centric)")
    cols = [
        c
        for c in ["game_id", "team", "opponent", "is_home", "win_probability"]
        if c in preds.columns
    ]
    st.dataframe(
        preds[cols].sort_values("win_probability", ascending=False),
        use_container_width=True,
    )

    if not odds.empty:
        st.subheader("Joined Odds + Predictions")
        merged = odds.merge(
            preds,
            on=["game_id", "team"],
            how="inner",
            suffixes=("_odds", "_pred"),
        )

        if "price" in merged.columns and "win_probability" in merged.columns:

            def american_to_implied(p):
                if p > 0:
                    return 100 / (p + 100)
                else:
                    return -p / (-p + 100)

            merged["implied_prob"] = merged["price"].apply(american_to_implied)
            merged["edge"] = merged["win_probability"] - merged["implied_prob"]

        display_cols = [
            c
            for c in [
                "game_id",
                "team",
                "price",
                "win_probability",
                "implied_prob",
                "edge",
            ]
            if c in merged.columns
        ]

        st.dataframe(
            merged[display_cols].sort_values(
                "edge", ascending=False, na_position="last"
            ),
            use_container_width=True,
        )


def backtest_tab():
    st.header("Backtest")

    st.markdown("Run a historical backtest over your predictions + odds + outcomes.")

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start date", value=None, key="bt_start")
    with col2:
        end_date = st.date_input("End date", value=None, key="bt_end")

    col3, col4, col5 = st.columns(3)
    with col3:
        starting_bankroll = st.number_input(
            "Starting bankroll", value=1000.0, min_value=0.0
        )
    with col4:
        min_edge = st.number_input(
            "Min edge to bet", value=0.03, min_value=0.0, max_value=1.0, step=0.01
        )
    with col5:
        kelly_fraction = st.number_input(
            "Kelly fraction", value=0.25, min_value=0.0, max_value=1.0, step=0.05
        )

    max_stake_fraction = st.slider(
        "Max stake as fraction of bankroll", 0.0, 1.0, 0.05, 0.01
    )

    if st.button("Run backtest"):
        cfg = BacktestConfig(
            starting_bankroll=starting_bankroll,
            min_edge=min_edge,
            kelly_fraction=kelly_fraction,
            max_stake_fraction=max_stake_fraction,
        )

        bt = Backtester(cfg)
        results = bt.run(
            start_date=start_date.isoformat() if start_date else None,
            end_date=end_date.isoformat() if end_date else None,
        )

        if not results:
            st.warning(
                "No results from backtest. Check that predictions, odds, and outcomes exist for this range."
            )
            return

        st.subheader("Summary metrics")
        st.write(f"Final bankroll: {results['final_bankroll']:.2f}")
        st.write(f"Total profit: {results['total_profit']:.2f}")
        st.write(f"ROI: {results['roi']:.3f}")
        st.write(f"Hit rate: {results['hit_rate']:.3f}")
        st.write(f"Max drawdown: {results['max_drawdown']:.3f}")

        records = results["records"]
        if not records.empty:
            st.subheader("Bankroll curve")
            fig = plot_bankroll_curve(records)
            st.pyplot(fig, clear_figure=True)

            st.subheader("Per-bet log")
            st.dataframe(
                records.sort_values("date"),
                use_container_width=True,
            )


def main():
    st.set_page_config(page_title="NBA Analytics & Betting Dashboard", layout="wide")

    tab1, tab2 = st.tabs(["Predictions", "Backtest"])

    with tab1:
        predictions_tab()
    with tab2:
        backtest_tab()


if __name__ == "__main__":
    main()
