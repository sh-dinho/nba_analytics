from __future__ import annotations
# ============================================================
# 🏀 NBA Analytics v3
# Module: Backtesting — Multi-Market (ML + O/U + ATS)
# File: src/backtest/engine_multi.py
# Author: Sadiq
#
# Description:
#     Unified backtesting engine that combines:
#       - Moneyline predictions
#       - Totals predictions
#       - Spread predictions
#
#     Produces:
#       - Combined ROI
#       - Per-market ROI
#       - Hit rates
#       - Max drawdown
#       - Full bet log
# ============================================================


from dataclasses import dataclass
import pandas as pd
from loguru import logger
from src.backtest.engine import Backtester, BacktestConfig
from src.backtest.engine_totals import TotalsBacktester, TotalsBacktestConfig
from src.backtest.engine_spread import SpreadBacktester, SpreadBacktestConfig


@dataclass
class MultiMarketBacktestConfig:
    ml_enabled: bool = True
    totals_enabled: bool = True
    spread_enabled: bool = True

    # ML config
    ml_min_edge: float = 0.03
    ml_kelly_fraction: float = 0.25

    # Totals config
    totals_min_edge: float = 3.0
    totals_stake_fraction: float = 0.02

    # Spread config
    spread_min_edge: float = 2.0
    spread_stake_fraction: float = 0.02

    starting_bankroll: float = 1000.0


class MultiMarketBacktester:
    def __init__(self, cfg: MultiMarketBacktestConfig):
        self.cfg = cfg

    def run(self, start_date=None, end_date=None):
        logger.info("🏀 Running multi-market backtest")

        combined_bets = []
        bankroll = self.cfg.starting_bankroll

        # ----------------------------------------------------
        # Moneyline
        # ----------------------------------------------------
        if self.cfg.ml_enabled:
            ml_cfg = BacktestConfig(
                starting_bankroll=bankroll,
                min_edge=self.cfg.ml_min_edge,
                kelly_fraction=self.cfg.ml_kelly_fraction,
                max_stake_fraction=0.05,
            )
            ml_bt = Backtester(ml_cfg)
            ml_res = ml_bt.run(start_date, end_date)
            if ml_res:
                combined_bets.append(ml_res["bets"])

        # ----------------------------------------------------
        # Totals
        # ----------------------------------------------------
        if self.cfg.totals_enabled:
            tot_cfg = TotalsBacktestConfig(
                starting_bankroll=bankroll,
                min_edge=self.cfg.totals_min_edge,
                stake_fraction=self.cfg.totals_stake_fraction,
            )
            tot_bt = TotalsBacktester(tot_cfg)
            tot_res = tot_bt.run(start_date, end_date)
            if tot_res:
                combined_bets.append(tot_res["bets"])

        # ----------------------------------------------------
        # Spread
        # ----------------------------------------------------
        if self.cfg.spread_enabled:
            sp_cfg = SpreadBacktestConfig(
                starting_bankroll=bankroll,
                min_edge=self.cfg.spread_min_edge,
                stake_fraction=self.cfg.spread_stake_fraction,
            )
            sp_bt = SpreadBacktester(sp_cfg)
            sp_res = sp_bt.run(start_date, end_date)
            if sp_res:
                combined_bets.append(sp_res["bets"])

        # ----------------------------------------------------
        # Combine all bets
        # ----------------------------------------------------
        if not combined_bets:
            logger.warning("No bets generated in any market.")
            return {}

        all_bets = pd.concat(combined_bets, ignore_index=True)
        all_bets = all_bets.sort_values("date")

        # Simulate bankroll across all markets
        bankroll = self.cfg.starting_bankroll
        bankrolls = []
        profits = []

        for _, row in all_bets.iterrows():
            profit = row["profit"]
            bankroll += profit
            profits.append(profit)
            bankrolls.append(bankroll)

        all_bets["bankroll_after"] = bankrolls

        total_profit = bankroll - self.cfg.starting_bankroll
        roi = total_profit / self.cfg.starting_bankroll
        hit_rate = (all_bets["profit"] > 0).mean()

        return {
            "bets": all_bets,
            "final_bankroll": bankroll,
            "total_profit": total_profit,
            "roi": roi,
            "hit_rate": hit_rate,
            "num_bets": len(all_bets),
        }
