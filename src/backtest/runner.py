from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Backtest Runner
# File: src/backtest/runner.py
# Author: Sadiq
#
# Description:
#     High-level wrapper for running backtests using the
#     canonical Backtester and BacktestConfig.
# ============================================================

from dataclasses import dataclass
import pandas as pd
from loguru import logger

from src.backtest.engine import Backtester, BacktestResult
from src.backtest.config import BacktestConfig


def run_backtest(
    df: pd.DataFrame,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """
    Run a full backtest on a canonical merged dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain:
            - win_probability
            - odds
            - closing_odds
            - edge
            - pick
            - actual_outcome
    config : BacktestConfig
        Optional bankroll + sizing configuration.

    Returns
    -------
    BacktestResult
        Full backtest metrics + bankroll curve + summary.
    """

    if config is None:
        config = BacktestConfig()

    if df.empty:
        logger.warning("Backtest: dataset is empty; returning zero metrics.")
        return BacktestResult(
            final_bankroll=config.starting_bankroll,
            total_profit=0.0,
            roi=0.0,
            hit_rate=0.0,
            max_drawdown=0.0,
            n_bets=0,
            avg_edge=0.0,
            clv=0.0,
            value_bets=pd.DataFrame(),
            bankroll_history=pd.DataFrame(),
            summary=pd.DataFrame(),
        )

    logger.info(f"Running backtest on {len(df)} rows")

    engine = Backtester(config)
    result = engine.run(df)

    logger.success(
        f"Backtest complete → ROI={result.roi:.2%}, "
        f"HitRate={result.hit_rate:.2%}, CLV={result.clv:.2%}, "
        f"Bets={result.n_bets}"
    )

    return result