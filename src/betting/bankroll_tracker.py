from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Bankroll Tracker
# File: src/betting/bankroll_tracker.py
# Author: Sadiq
#
# Description:
#     Computes bankroll over time using settled bets.
#     Version‑agnostic and aligned with the modern betting stack.
# ============================================================

import pandas as pd
from loguru import logger

from src.config.betting import BANKROLL
from src.config.paths import BET_LOG_PATH


def compute_bankroll_curve(
    initial_bankroll: float = BANKROLL,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Compute bankroll over time using settled bets.

    Returns a DataFrame with:
        timestamp, game_id, stake, profit, bankroll, drawdown

    Parameters
    ----------
    initial_bankroll : float
        Starting bankroll.
    start_date : str | None
        Optional ISO date filter.
    end_date : str | None
        Optional ISO date filter.
    """

    if not BET_LOG_PATH.exists():
        logger.warning("bankroll_tracker(): no bet log found.")
        return pd.DataFrame()

    df = pd.read_csv(BET_LOG_PATH)

    # Must have profit column (settled bets)
    if "profit" not in df.columns:
        logger.warning("bankroll_tracker(): bets not settled yet.")
        return pd.DataFrame()

    # Clean + sort
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp")

    # Optional date filtering
    if start_date:
        df = df[df["timestamp"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["timestamp"] <= pd.to_datetime(end_date)]

    if df.empty:
        logger.warning("bankroll_tracker(): no rows after filtering.")
        return pd.DataFrame()

    df["profit"] = df["profit"].fillna(0.0)

    bankroll = initial_bankroll
    curve = []

    for _, row in df.iterrows():
        bankroll += row["profit"]
        curve.append(
            {
                "timestamp": row["timestamp"],
                "game_id": row.get("game_id"),
                "stake": row.get("stake", 0.0),
                "profit": row["profit"],
                "bankroll": bankroll,
            }
        )

    curve_df = pd.DataFrame(curve)

    # Compute drawdown
    curve_df["peak"] = curve_df["bankroll"].cummax()
    curve_df["drawdown"] = (curve_df["bankroll"] - curve_df["peak"]) / curve_df["peak"]

    return curve_df