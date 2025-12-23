from __future__ import annotations
# ============================================================
# 🏀 NBA Analytics v3
# Module: Dashboard — Advanced Predictions (ML + O/U + ATS)
# File: src/dashboard/tabs/advanced_predictions.py
# Author: Sadiq
#
# Description:
#     Unified view of:
#       - Moneyline predictions
#       - Totals predictions
#       - Spread predictions
#       - Market lines (totals + spread)
#       - Edges + recommended bets
#
#     This tab is the core of the multi-market prediction suite.
# ============================================================


import pandas as pd
import streamlit as st

from src.config.paths import (
    PREDICTIONS_DIR,
    DATA_DIR,
    ODDS_DIR,
)

TOTALS_DIR = DATA_DIR / "predictions_totals"
SPREAD_DIR = DATA_DIR / "predictions_spread"


# ------------------------------------------------------------
# Loaders
# ------------------------------------------------------------
def load_ml(pred_date):
    path = PREDICTIONS_DIR / f"predictions_{pred_date}.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def load_totals(pred_date):
    path = TOTALS_DIR / f"totals_{pred_date}.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def load_spread(pred_date):
    path = SPREAD_DIR / f"spread_{pred_date}.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def load_odds(pred_date):
    path = ODDS_DIR / f"odds_{pred_date}.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


# ------------------------------------------------------------
# Edge calculations
# ------------------------------------------------------------
def implied_prob_from_american(price):
    if price > 0:
        return 100 / (price + 100)
    return -price / (-price + 100)


def compute_totals_edges(df):
    df = df.copy()
    df["edge_over"] = df["predicted_total"] - df["market_total"]
    df["edge_under"] = df["market_total"] - df["predicted_total"]
    df["recommendation"] = df.apply(
        lambda r: "OVER" if r["edge_over"] > r["edge_under"] else "UNDER",
        axis=1,
    )
    df["edge"] = df[["edge_over", "edge_under"]].max(axis=1)
    return df


def compute_spread_edges(df):
    df = df.copy()
    df["edge_home"] = df["predicted_margin"] - df["market_spread"]
    df["edge_away"] = df["market_spread"] - df["predicted_margin"]
    df["recommendation"] = df.apply(
        lambda r: "HOME ATS" if r["edge_home"] > r["edge_away"] else "AWAY ATS",
        axis=1,
    )
    df["edge"] = df[["edge_home", "edge_away"]].max(axis=1)
    return df


# ------------------------------------------------------------
# Main tab renderer
# ------------------------------------------------------------
def render_advanced_predictions_tab():
    st.header("Advanced Predictions (Moneyline + Totals + Spread)")

    pred_date = st.date_input("Prediction date")

    ml = load_ml(pred_date)
    totals = load_totals(pred_date)
    spread = load_spread(pred_date)
    odds = load_odds(pred_date)

    if ml.empty and totals.empty and spread.empty:
        st.warning("No predictions available for this date.")
        return

    # --------------------------------------------------------
    # Moneyline
    # --------------------------------------------------------
    st.subheader("Moneyline Predictions")
    if not ml.empty:
        ml_display = ml[["game_id", "team", "opponent", "win_probability"]]
        st.dataframe(
            ml_display.sort_values("win_probability", ascending=False),
            use_container_width=True,
        )
    else:
        st.info("No moneyline predictions found.")

    # --------------------------------------------------------
    # Totals (O/U)
    # --------------------------------------------------------
    st.subheader("Totals (Over/Under) Predictions")

    if not totals.empty and not odds.empty:
        # Join totals with market totals
        ou = totals.merge(
            odds[["game_id", "market_total"]],
            on="game_id",
            how="left",
        )

        ou = compute_totals_edges(ou)

        st.dataframe(
            ou[
                [
                    "game_id",
                    "home_team",
                    "away_team",
                    "predicted_total",
                    "market_total",
                    "edge",
                    "recommendation",
                ]
            ].sort_values("edge", ascending=False),
            use_container_width=True,
        )
    else:
        st.info("Totals predictions or market totals not available.")

    # --------------------------------------------------------
    # Spread (ATS)
    # --------------------------------------------------------
    st.subheader("Spread (ATS) Predictions")

    if not spread.empty and not odds.empty:
        ats = spread.merge(
            odds[["game_id", "market_spread"]],
            on="game_id",
            how="left",
        )

        ats = compute_spread_edges(ats)

        st.dataframe(
            ats[
                [
                    "game_id",
                    "home_team",
                    "away_team",
                    "predicted_margin",
                    "market_spread",
                    "edge",
                    "recommendation",
                ]
            ].sort_values("edge", ascending=False),
            use_container_width=True,
        )
    else:
        st.info("Spread predictions or market spreads not available.")

    # --------------------------------------------------------
    # Combined Recommendations
    # --------------------------------------------------------
    st.subheader("Unified Recommendations")

    combined = []

    if not ml.empty:
        for _, r in ml.iterrows():
            combined.append(
                {
                    "game_id": r["game_id"],
                    "market": "Moneyline",
                    "team": r["team"],
                    "recommendation": f"{r['team']} ML",
                    "confidence": r["win_probability"],
                }
            )

    if not totals.empty and not odds.empty:
        for _, r in ou.iterrows():
            combined.append(
                {
                    "game_id": r["game_id"],
                    "market": "Totals",
                    "team": f"{r['home_team']} vs {r['away_team']}",
                    "recommendation": r["recommendation"],
                    "confidence": abs(r["edge"]),
                }
            )

    if not spread.empty and not odds.empty:
        for _, r in ats.iterrows():
            combined.append(
                {
                    "game_id": r["game_id"],
                    "market": "Spread",
                    "team": f"{r['home_team']} vs {r['away_team']}",
                    "recommendation": r["recommendation"],
                    "confidence": abs(r["edge"]),
                }
            )

    if combined:
        df_combined = pd.DataFrame(combined)
        st.dataframe(
            df_combined.sort_values("confidence", ascending=False),
            use_container_width=True,
        )
    else:
        st.info("No combined recommendations available.")
