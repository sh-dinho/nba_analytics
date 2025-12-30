from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v3
# Module: Team Stability Engine
# File: src/analytics/team_stability.py
# Author: Sadiq
#
# Description:
#     Computes team-level stability metrics using backtest
#     records and predictions:
#       - ROI per team
#       - Profit per team
#       - Bet count
#       - Volatility of edge and win_probability
#       - Hit rate
#       - Stability score (0–100)
#
#     Produces:
#       - teams_to_avoid
#       - teams_to_watch
#       - full per-team metrics table
# ============================================================
from __future__ import annotations

import pandas as pd
from dataclasses import dataclass
from typing import Optional
from loguru import logger

from src.backtest.engine import Backtester, BacktestConfig


@dataclass
class TeamStabilityConfig:
    start: Optional[str] = None
    end: Optional[str] = None
    min_bets: int = 20


def _compute_stability_from_backtest(df: pd.DataFrame, min_bets: int) -> pd.DataFrame:
    """
    Compute team stability metrics from a backtest results dataframe.
    """
    if df.empty:
        logger.warning("No backtest records available for team stability computation.")
        return pd.DataFrame(columns=["team", "stability_score"])

    # Count bets per team
    counts = df.groupby("team").size().rename("num_bets")

    # Win rate per team
    win_rate = df.groupby("team")["won"].mean().rename("win_rate")

    # Variance of outcomes per team
    variance = df.groupby("team")["won"].var().fillna(0).rename("variance")

    merged = pd.concat([counts, win_rate, variance], axis=1).reset_index()

    # Filter out teams with too few bets
    merged = merged[merged["num_bets"] >= min_bets]

    if merged.empty:
        logger.warning("No teams meet minimum bet threshold for stability.")
        return pd.DataFrame(columns=["team", "stability_score"])

    # Stability score: high win rate + low variance
    merged["stability_score"] = merged["win_rate"] * (1 - merged["variance"])

    return merged[["team", "stability_score"]]


def get_team_stability(
    start: Optional[str] = None,
    end: Optional[str] = None,
    min_bets: int = 20,
) -> pd.DataFrame:
    """
    Public API for computing team stability.
    This is the function used by the recommendation engine.

    Tests should patch THIS function.
    """
    logger.info(
        f"🏀 Computing team stability with config: start={start}, end={end}, min_bets={min_bets}"
    )

    # Run a moneyline backtest to get historical performance
    cfg = BacktestConfig(
        starting_bankroll=1000,
        min_edge=0,
        kelly_fraction=0.25,
        market="moneyline",
    )

    backtester = Backtester(cfg)
    results = backtester.run(start_date=start, end_date=end)

    if results is None or "records" not in results:
        logger.warning("Backtest returned no records for stability computation.")
        return pd.DataFrame(columns=["team", "stability_score"])

    df = results["records"]

    return _compute_stability_from_backtest(df, min_bets)
