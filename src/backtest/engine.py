# ============================================================
# Project: NBA Analytics & Betting Engine
# Module: Backtesting Engine
# Author: Sadiq
#
# Description:
#     This module provides a full historical backtesting engine
#     for evaluating model-driven betting strategies.
#
#     It loads:
#       - Historical model predictions
#       - Historical bookmaker odds
#       - Actual game outcomes (from canonical long snapshot)
#
#     It joins these into a unified dataset and simulates:
#       - Edge calculation
#       - Kelly-based stake sizing
#       - Bet execution and bankroll evolution
#       - Profit/loss tracking
#       - ROI, hit rate, max drawdown
#
#     The engine is fully modular:
#       - BacktestConfig controls strategy parameters
#       - Backtester.run() executes the simulation
#       - Results include per-bet logs and bankroll curves
#
#     This module integrates with:
#       - The orchestrator (for daily runs)
#       - The dashboard (for visualization)
#       - Telegram notifications (for charts and summaries)
#
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.config.paths import PREDICTIONS_DIR, ODDS_DIR, LONG_SNAPSHOT


# ============================================================
# Configuration
# ============================================================


@dataclass
class BacktestConfig:
    starting_bankroll: float = 1000.0
    min_edge: float = 0.03  # minimum edge required to place a bet
    kelly_fraction: float = 0.25  # fractional Kelly
    max_stake_fraction: float = 0.05  # cap stake to 5% of bankroll
    market: str = "moneyline"  # placeholder for future markets


# ============================================================
# Utility Functions
# ============================================================


def american_to_implied(price: float) -> float:
    """Convert American odds to implied probability."""
    if price > 0:
        return 100 / (price + 100)
    else:
        return -price / (-price + 100)


def kelly_fraction(win_prob: float, price: float) -> float:
    """
    Kelly formula for American odds.
    Convert to decimal odds first.
    """
    if price > 0:
        decimal_odds = 1 + price / 100
    else:
        decimal_odds = 1 + 100 / -price

    b = decimal_odds - 1
    p = win_prob
    q = 1 - p

    k = (b * p - q) / b
    return max(0.0, k)


# ============================================================
# Backtester
# ============================================================


class Backtester:
    def __init__(self, config: BacktestConfig):
        self.config = config

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def run(
        self, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> dict:
        """
        Run backtest over predictions + odds + outcomes for the given date range.
        """
        df = self._load_joined_data(start_date, end_date)

        if df.empty:
            logger.warning("No data available for backtest in given date range.")
            return {}

        df = self._compute_edges(df)
        bets = self._apply_strategy(df)
        results = self._simulate_bankroll(bets)

        return results

    # --------------------------------------------------------
    # Data Loading & Joining
    # --------------------------------------------------------

    def _load_joined_data(
        self, start_date: Optional[str], end_date: Optional[str]
    ) -> pd.DataFrame:
        """
        Load predictions, odds, and outcomes, join into a single DataFrame.
        """
        # 1) Load outcomes
        long_df = pd.read_parquet(LONG_SNAPSHOT)
        long_df["date"] = pd.to_datetime(long_df["date"]).dt.date
        long_df = long_df[["game_id", "team", "won", "date"]]

        # 2) Load predictions
        pred_files = sorted(PREDICTIONS_DIR.glob("predictions_*.parquet"))
        preds_list = []

        for path in pred_files:
            dt_str = path.stem.replace("predictions_", "")
            dt = pd.to_datetime(dt_str).date()

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

        # 3) Load odds
        odds_files = sorted(ODDS_DIR.glob("odds_*.parquet"))
        odds_list = []

        for path in odds_files:
            dt_str = path.stem.replace("odds_", "")
            dt = pd.to_datetime(dt_str).date()

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

        # 4) Join predictions + odds
        joined = odds.merge(
            preds,
            on=["game_id", "team", "date"],
            how="inner",
            suffixes=("_odds", "_pred"),
        )

        # 5) Join outcomes
        joined = joined.merge(
            long_df,
            on=["game_id", "team"],
            how="left",
        )

        # Remove rows without outcomes (future games)
        joined = joined.dropna(subset=["won"])

        return joined

    # --------------------------------------------------------
    # Strategy Logic
    # --------------------------------------------------------

    def _compute_edges(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["implied_prob"] = df["price"].apply(american_to_implied)
        df["edge"] = df["win_probability"] - df["implied_prob"]
        return df

    def _apply_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply edge + fractional Kelly strategy.
        """
        df = df.copy()
        bankroll = self.config.starting_bankroll

        stakes = []

        for _, row in df.iterrows():
            edge = row["edge"]

            if edge <= self.config.min_edge:
                stakes.append(0.0)
                continue

            k_full = kelly_fraction(row["win_probability"], row["price"])
            k_used = k_full * self.config.kelly_fraction

            stake = bankroll * k_used
            stake = min(stake, bankroll * self.config.max_stake_fraction)

            stakes.append(stake)

        df["stake"] = stakes
        df = df[df["stake"] > 0].copy()

        return df

    # --------------------------------------------------------
    # Bankroll Simulation
    # --------------------------------------------------------

    def _simulate_bankroll(self, bets: pd.DataFrame) -> dict:
        if bets.empty:
            logger.warning("No bets placed under the current strategy.")
            return {}

        bets = bets.sort_values("date").reset_index(drop=True)

        bankroll = self.config.starting_bankroll
        bankrolls = []
        profits = []
        daily_records = []

        for _, row in bets.iterrows():
            stake = row["stake"]
            price = row["price"]
            won = row["won"]

            if stake <= 0 or bankroll <= 0:
                bankrolls.append(bankroll)
                profits.append(0.0)
                continue

            # Compute profit
            if price > 0:
                profit = stake * (price / 100) if won == 1 else -stake
            else:
                profit = stake * (100 / -price) if won == 1 else -stake

            bankroll += profit
            bankrolls.append(bankroll)
            profits.append(profit)

            daily_records.append(
                {
                    "date": row["date"],
                    "game_id": row["game_id"],
                    "team": row["team"],
                    "price": price,
                    "win_probability": row["win_probability"],
                    "implied_prob": row["implied_prob"],
                    "edge": row["edge"],
                    "stake": stake,
                    "profit": profit,
                    "bankroll_after": bankroll,
                }
            )

        bets["profit"] = profits
        bets["bankroll_after"] = bankrolls

        total_profit = bankroll - self.config.starting_bankroll
        roi = total_profit / self.config.starting_bankroll
        hit_rate = (bets["profit"] > 0).mean()
        max_drawdown = self._max_drawdown([r["bankroll_after"] for r in daily_records])

        return {
            "config": self.config,
            "bets": bets,
            "records": pd.DataFrame(daily_records),
            "final_bankroll": bankroll,
            "total_profit": total_profit,
            "roi": roi,
            "hit_rate": hit_rate,
            "max_drawdown": max_drawdown,
        }

    @staticmethod
    def _max_drawdown(series) -> float:
        series = np.array(series, dtype=float)
        running_max = np.maximum.accumulate(series)
        drawdowns = (series - running_max) / running_max
        return float(drawdowns.min())
