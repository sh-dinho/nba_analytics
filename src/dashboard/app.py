# ============================================================
# 🏀 NBA Analytics v3
# Module: Dashboard App
# File: src/dashboard/app.py
# Author: Sadiq
#
# Description:
#     Streamlit-based client portal for NBA Analytics v3:
#       - Predictions tab
#       - Backtest / What-if simulator
#       - Accuracy tab
#       - Strategy Comparison (admin)
#       - Model Leaderboard (admin)
#
#     Integrates with:
#       - Model registry (index.json)
#       - Backtesting engine
#       - Accuracy engine
#       - Reports generator
#       - Auth module for role-based views
# ============================================================

from __future__ import annotations

from datetime import date

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.config.paths import PREDICTIONS_DIR, ODDS_DIR
from src.backtest.engine import Backtester, BacktestConfig
from src.backtest.accuracy import AccuracyEngine
from src.backtest.compare import compare_strategies
from src.dashboard.auth import require_login, is_admin
from src.reports.backtest_report import generate_backtest_accuracy_report
from src.dashboard.tabs.model_leaderboard import render_model_leaderboard
from src.dashboard.tabs.advanced_predictions import render_advanced_predictions_tab
from src.dashboard.tabs.team_stability import render_team_stability_tab

# ============================================================
# Helpers: data loading & plotting
# ============================================================


def _load_predictions(pred_date: date) -> pd.DataFrame:
    path = PREDICTIONS_DIR / f"predictions_{pred_date}.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def _load_odds(pred_date: date) -> pd.DataFrame:
    path = ODDS_DIR / f"odds_{pred_date}.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def plot_bankroll_curve(records: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4))
    records = records.sort_values("date")
    ax.plot(records["date"], records["bankroll_after"], marker="o")
    ax.set_title("Bankroll Over Time")
    ax.set_ylabel("Bankroll")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


# ============================================================
# Tab: Predictions
# ============================================================


def render_predictions_tab():
    st.header("Predictions")

    pred_date = st.date_input("Prediction date", value=date.today(), key="pred_date")

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

    # Show model metadata if available
    meta_cols = [
        c
        for c in ["model_name", "model_version", "feature_version"]
        if c in preds.columns
    ]
    if meta_cols:
        meta = preds[meta_cols].drop_duplicates().head(1)
        st.markdown("**Model Metadata**")
        st.json(meta.to_dict(orient="records")[0])

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
                "model_name",
                "model_version",
            ]
            if c in merged.columns
        ]

        st.dataframe(
            merged[display_cols].sort_values(
                "edge", ascending=False, na_position="last"
            ),
            use_container_width=True,
        )


# ============================================================
# Tab: Backtest / What-if Simulator
# ============================================================


def render_backtest_tab():
    st.header("Backtest / What-if Simulator")

    st.markdown(
        "_Adjust the parameters below to see how different bankroll and risk "
        "settings would have performed on historical data._"
    )

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

    # Optional: filter by model
    st.subheader("Model filter (optional)")
    model_name = st.text_input("Model name (exact match, optional)")
    model_version = st.text_input("Model version (timestamp, optional)")

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
            model_name=model_name or None,
            model_version=model_version or None,
        )

        if not results:
            st.warning(
                "No results from backtest. Check that predictions, odds, and outcomes "
                "exist for this range and (optional) model selection."
            )
            return

        st.subheader("Summary metrics")
        st.write(f"Final bankroll: {results['final_bankroll']:.2f}")
        st.write(f"Total profit: {results['total_profit']:.2f}")
        st.write(f"ROI: {results['roi']:.3f}")
        st.write(f"Hit rate: {results['hit_rate']:.3f}")
        st.write(f"Max drawdown: {results['max_drawdown']:.3f}")
        st.write(
            f"Bets: {results['num_bets']}, Wins: {results['num_wins']}, "
            f"Losses: {results['num_losses']}, Pushes: {results['num_pushes']}"
        )

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

        if st.button("Generate client-ready report"):
            report_path = generate_backtest_accuracy_report(
                start_date=start_date.isoformat() if start_date else None,
                end_date=end_date.isoformat() if end_date else None,
                config=cfg,
                decision_threshold=0.5,
            )
            st.success(f"Report generated: {report_path}")
            st.markdown(f"[Download report]({report_path})")


# ============================================================
# Tab: Accuracy
# ============================================================


def render_accuracy_tab():
    st.header("Model Accuracy")

    threshold = st.slider(
        "Decision threshold (win_probability >= threshold => predict win)",
        0.0,
        1.0,
        0.5,
        0.01,
    )
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start date (optional)", value=None, key="acc_start")
    with col2:
        end_date = st.date_input("End date (optional)", value=None, key="acc_end")

    if st.button("Compute accuracy"):
        engine = AccuracyEngine(threshold=threshold)
        res = engine.run(
            start_date=start_date.isoformat() if start_date else None,
            end_date=end_date.isoformat() if end_date else None,
        )

        if res.total_examples == 0:
            st.warning("No data for accuracy computation in this range.")
            return

        st.subheader("Overall")
        st.write(f"Overall accuracy: {res.overall_accuracy:.3f}")
        st.write(f"Total examples: {res.total_examples}")

        if not res.by_season.empty:
            st.subheader("By season")
            st.dataframe(res.by_season, use_container_width=True)

        st.subheader("Sample predictions vs outcomes")
        sample_cols = [
            "date",
            "game_id",
            "team",
            "win_probability",
            "won",
            "predicted_win",
            "correct",
        ]
        sample = res.raw[sample_cols].sort_values("date", ascending=False).head(200)
        st.dataframe(sample, use_container_width=True)


# ============================================================
# Tab: Strategy Comparison (admin)
# ============================================================


def render_strategy_comparison_tab():
    st.header("Strategy Comparison")

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start date", value=None, key="cmp_start")
    with col2:
        end_date = st.date_input("End date", value=None, key="cmp_end")

    st.markdown("Define a few strategies to compare:")

    col3, col4, col5 = st.columns(3)
    with col3:
        min_edge_base = st.number_input("Baseline min edge", value=0.0, step=0.01)
    with col4:
        min_edge_conservative = st.number_input(
            "Main strategy min edge", value=0.03, step=0.01
        )
    with col5:
        min_edge_aggressive = st.number_input(
            "Aggressive min edge", value=0.05, step=0.01
        )

    col6, col7 = st.columns(2)
    with col6:
        kelly_fraction_base = st.number_input(
            "Baseline Kelly fraction", value=0.0, step=0.05
        )
    with col7:
        kelly_fraction_main = st.number_input(
            "Main / Aggressive Kelly fraction", value=0.25, step=0.05
        )

    if st.button("Compare strategies"):
        configs = {
            "Flat no-edge baseline": BacktestConfig(
                starting_bankroll=1000.0,
                min_edge=min_edge_base,
                kelly_fraction=kelly_fraction_base,
                max_stake_fraction=0.02,
            ),
            "Main strategy": BacktestConfig(
                starting_bankroll=1000.0,
                min_edge=min_edge_conservative,
                kelly_fraction=kelly_fraction_main,
                max_stake_fraction=0.05,
            ),
            "Aggressive edge strategy": BacktestConfig(
                starting_bankroll=1000.0,
                min_edge=min_edge_aggressive,
                kelly_fraction=kelly_fraction_main,
                max_stake_fraction=0.1,
            ),
        }

        df = compare_strategies(
            configs,
            start_date=start_date.isoformat() if start_date else None,
            end_date=end_date.isoformat() if end_date else None,
        )

        if df.empty or not df["has_data"].any():
            st.warning("No data for comparison in this range.")
            return

        st.subheader("Strategy comparison table")
        st.dataframe(
            df.sort_values("roi", ascending=False),
            use_container_width=True,
        )


# ============================================================
# Main app
# ============================================================


def main():
    st.set_page_config(page_title="NBA Analytics v3 — Client Portal", layout="wide")

    require_login()

    st.sidebar.write(f"Logged in as: **{st.session_state['username']}**")
    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.experimental_rerun()

    # Tabs: admin gets everything, client gets a subset
    if is_admin():
        tab_names = [
            "Predictions",
            "Advanced Predictions",
            "Backtest / What-if",
            "Accuracy",
            "Strategy Comparison",
            "Model Leaderboard",
            "Team Stability",
        ]
    else:
        tab_names = [
            "Predictions",
            "Advanced Predictions",
            "Backtest / What-if",
            "Accuracy",
        ]

    name_to_renderer = {
        "Predictions": render_predictions_tab,
        "Advanced Predictions": render_advanced_predictions_tab,
        "Backtest / What-if": render_backtest_tab,
        "Accuracy": render_accuracy_tab,
        "Strategy Comparison": render_strategy_comparison_tab,
        "Model Leaderboard": render_model_leaderboard,
        "Team Stability": render_team_stability_tab,
    }

    for tab, name in zip(st.tabs, tab_names):
        with tab:
            renderer = name_to_renderer[name]
            renderer()


if __name__ == "__main__":
    main()
