from __future__ import annotations
# ============================================================
# 🏀 NBA Analytics v3
# Module: Strategy Comparison
# File: src/backtest/compare.py
# Author: Sadiq
#
# Description:
#     Runs multiple backtest configurations and compares:
#       - ROI
#       - final bankroll
#       - hit rate
#       - drawdown
#       - bet volume
# ============================================================


import pandas as pd

from src.backtest.engine import Backtester


def compare_strategies(configs: dict, start_date: str, end_date: str) -> pd.DataFrame:
    rows = []

    for name, cfg in configs.items():
        bt = Backtester(cfg)
        res = bt.run(start_date=start_date, end_date=end_date)

        if not res:
            rows.append(
                {
                    "strategy": name,
                    "roi": None,
                    "final_bankroll": None,
                    "hit_rate": None,
                    "max_drawdown": None,
                    "num_bets": 0,
                    "has_data": False,
                }
            )
            continue

        rows.append(
            {
                "strategy": name,
                "roi": res["roi"],
                "final_bankroll": res["final_bankroll"],
                "hit_rate": res["hit_rate"],
                "max_drawdown": res["max_drawdown"],
                "num_bets": res["num_bets"],
                "has_data": True,
            }
        )

    return pd.DataFrame(rows)
