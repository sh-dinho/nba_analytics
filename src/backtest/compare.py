from __future__ import annotations
# ============================================================
# Project: NBA Analytics & Betting Engine
# Module: Strategy Comparison
# Author: Sadiq
#
# Description:
#     Run multiple backtest configurations over the same range
#     and return a comparison table for consulting/demo.
# ============================================================


import pandas as pd

from src.backtest.engine import Backtester, BacktestConfig


def compare_strategies(
    configs: dict[str, BacktestConfig],
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    rows = []

    for name, cfg in configs.items():
        bt = Backtester(cfg)
        res = bt.run(start_date=start_date, end_date=end_date)

        if not res:
            rows.append(
                {
                    "strategy": name,
                    "has_data": False,
                    "final_bankroll": None,
                    "total_profit": None,
                    "roi": None,
                    "hit_rate": None,
                    "max_drawdown": None,
                    "num_bets": 0,
                }
            )
            continue

        rows.append(
            {
                "strategy": name,
                "has_data": True,
                "final_bankroll": res["final_bankroll"],
                "total_profit": res["total_profit"],
                "roi": res["roi"],
                "hit_rate": res["hit_rate"],
                "max_drawdown": res["max_drawdown"],
                "num_bets": res["num_bets"],
                "min_edge": cfg.min_edge,
                "kelly_fraction": cfg.kelly_fraction,
                "max_stake_fraction": cfg.max_stake_fraction,
            }
        )

    df = pd.DataFrame(rows)
    return df
