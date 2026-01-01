from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Strategy Comparison
# File: src/backtest/compare.py
# Author: Sadiq
#
# Description:
#     Compare multiple backtest configurations using the
#     canonical Backtester + BacktestResult.
#
#     Metrics:
#       • ROI
#       • final bankroll
#       • hit rate
#       • max drawdown
#       • bet volume
#       • average edge
#       • CLV
# ============================================================

import pandas as pd
from loguru import logger

from src.backtest.engine import Backtester
from src.backtest.config import BacktestConfig


def compare_strategies(
    configs: dict[str, BacktestConfig],
    merged_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare multiple backtest configurations on the same dataset.

    Parameters
    ----------
    configs : dict[str, BacktestConfig]
        Mapping: strategy name → BacktestConfig
    merged_df : pd.DataFrame
        Canonical merged dataset produced by load_backtest_data()

    Returns
    -------
    pd.DataFrame
        Summary of performance for each strategy.
    """

    logger.info("🏀 Running strategy comparison across configurations...")

    if merged_df.empty:
        logger.warning("compare_strategies(): merged_df is empty.")
        return pd.DataFrame()

    rows = []

    for name, cfg in configs.items():
        logger.info(f"→ Running strategy '{name}'")

        bt = Backtester(cfg)
        result = bt.run(merged_df)

        if result is None or result.n_bets == 0:
            logger.warning(f"Strategy '{name}' returned no bets.")
            rows.append(
                {
                    "strategy": name,
                    "roi": None,
                    "final_bankroll": None,
                    "hit_rate": None,
                    "max_drawdown": None,
                    "num_bets": 0,
                    "avg_edge": None,
                    "clv": None,
                    "has_data": False,
                }
            )
            continue

        rows.append(
            {
                "strategy": name,
                "roi": result.roi,
                "final_bankroll": result.final_bankroll,
                "hit_rate": result.hit_rate,
                "max_drawdown": result.max_drawdown,
                "num_bets": result.n_bets,
                "avg_edge": result.avg_edge,
                "clv": result.clv,
                "has_data": True,
            }
        )

    df = pd.DataFrame(rows)
    return df.sort_values("roi", ascending=False).reset_index(drop=True)