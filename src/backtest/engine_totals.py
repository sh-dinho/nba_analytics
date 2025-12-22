# ============================================================
# 🏀 NBA Analytics v3
# Module: Backtesting — Totals (Over/Under)
# File: src/backtest/engine_totals.py
# Author: Sadiq
#
# Description:
#     Backtests Over/Under predictions using:
#       - predicted_total
#       - market_total
#       - actual total points
#
#     Computes:
#       - ROI
#       - hit rate
#       - max drawdown
#       - per-bet logs
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.config.paths import (
    LONG_SNAPSHOT,
    DATA_DIR,
    ODDS_DIR,
)

TOTALS_PRED_DIR = DATA_DIR / "predictions_totals"


@dataclass
class TotalsBacktestConfig:
    starting_bankroll: float = 1000.0
    min_edge: float = 3.0  # points
    stake_fraction: float = 0.02  # flat fraction of bankroll per bet


class TotalsBacktester:
    def __init__(self, cfg: TotalsBacktestConfig):
        self.cfg = cfg

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------
    def run(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> dict:
        logger.info(f"🏀 Running totals backtest start={start_date}, end={end_date}")

        df = self._load_joined_data(start_date, end_date)
        if df.empty:
            logger.warning("No totals data available for backtest.")
            return {}

        df = self._compute_edges(df)
        bets = self._apply_strategy(df)
        results = self._simulate_bankroll(bets)

        logger.success(
            f"[Totals] Backtest complete. Final bankroll={results['final_bankroll']:.2f}, "
            f"ROI={results['roi']:.3f}, bets={results['num_bets']}"
        )

        return results

    # --------------------------------------------------------
    # Data loading
    # --------------------------------------------------------
    def _load_joined_data(self, start_date, end_date):
        # Load predictions
        pred_files = sorted(TOTALS_PRED_DIR.glob("totals_*.parquet"))
        preds_list = []

        for path in pred_files:
            dt = path.stem.replace("totals_", "")
            dt = pd.to_datetime(dt).date()

            if start_date and dt < pd.to_datetime(start_date).date():
                continue
            if end_date and dt > pd.to_datetime(end_date).date():
                continue

            df = pd.read_parquet(path)
            df["date"] = dt
            preds_list.append(df)

        if not preds_list:
            return pd.DataFrame()

        preds = pd.concat(preds_list, ignore_index=True)

        # Load odds
        odds_files = sorted(ODDS_DIR.glob("odds_*.parquet"))
        odds_list = []

        for path in odds_files:
            dt = path.stem.replace("odds_", "")
            dt = pd.to_datetime(dt).date()

            if start_date and dt < pd.to_datetime(start_date).date():
                continue
            if end_date and dt > pd.to_datetime(end_date).date():
                continue

            df = pd.read_parquet(path)
            df["date"] = dt
            odds_list.append(df)

        if not odds_list:
            return pd.DataFrame()

        odds = pd.concat(odds_list, ignore_index=True)

        # Load outcomes
        long_df = pd.read_parquet(LONG_SNAPSHOT)
        long_df["date"] = pd.to_datetime(long_df["date"]).dt.date

        # Compute actual totals (home rows only)
        home_rows = long_df[long_df["is_home"] == True].copy()
        home_rows["actual_total"] = (
            home_rows["points_for"] + home_rows["points_against"]
        )

        # Join predictions + odds + outcomes
        joined = preds.merge(
            odds[["game_id", "market_total"]],
            on="game_id",
            how="inner",
        ).merge(
            home_rows[["game_id", "date", "actual_total"]],
            on=["game_id", "date"],
            how="inner",
        )

        return joined

    # --------------------------------------------------------
    # Strategy
    # --------------------------------------------------------
    def _compute_edges(self, df):
        df = df.copy()
        df["edge_over"] = df["predicted_total"] - df["market_total"]
        df["edge_under"] = df["market_total"] - df["predicted_total"]
        df["edge"] = df[["edge_over", "edge_under"]].max(axis=1)
        df["direction"] = df.apply(
            lambda r: "OVER" if r["edge_over"] > r["edge_under"] else "UNDER",
            axis=1,
        )
        return df

    def _apply_strategy(self, df):
        df = df.copy()
        df = df[df["edge"] >= self.cfg.min_edge]

        if df.empty:
            return df

        df["stake"] = self.cfg.starting_bankroll * self.cfg.stake_fraction
        return df

    # --------------------------------------------------------
    # Bankroll simulation
    # --------------------------------------------------------
    def _simulate_bankroll(self, bets):
        if bets.empty:
            return {}

        bankroll = self.cfg.starting_bankroll
        bankrolls = []
        profits = []

        for _, row in bets.iterrows():
            stake = row["stake"]
            actual = row["actual_total"]
            market = row["market_total"]
            direction = row["direction"]

            if direction == "OVER":
                won = actual > market
            else:
                won = actual < market

            profit = stake if won else -stake
            bankroll += profit

            profits.append(profit)
            bankrolls.append(bankroll)

        bets = bets.copy()
        bets["profit"] = profits
        bets["bankroll_after"] = bankrolls

        total_profit = bankroll - self.cfg.starting_bankroll
        roi = total_profit / self.cfg.starting_bankroll
        hit_rate = (bets["profit"] > 0).mean()

        return {
            "bets": bets,
            "final_bankroll": bankroll,
            "total_profit": total_profit,
            "roi": roi,
            "hit_rate": hit_rate,
            "num_bets": len(bets),
        }
