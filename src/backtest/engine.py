from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Backtesting Engine
# File: src/backtest/engine.py
# Author: Sadiq
#
# Description:
#     Version‑agnostic backtesting engine supporting:
#       • bankroll simulation
#       • fractional Kelly sizing
#       • min-edge filtering
#       • max stake fraction
#       • CLV
#       • value bet extraction
#       • bankroll curve
# ============================================================

from dataclasses import dataclass, asdict
from typing import Any, Dict

import numpy as np
import pandas as pd
from loguru import logger

from src.backtest.config import BacktestConfig


@dataclass
class BacktestResult:
    final_bankroll: float
    total_profit: float
    roi: float
    accuracy: float
    max_drawdown: float
    n_bets: int
    avg_edge: float
    clv: float

    # Rich outputs (not serialized in detail by default)
    value_bets: pd.DataFrame
    bankroll_history: pd.DataFrame
    summary: pd.DataFrame

    def to_dict(self) -> Dict[str, Any]:
        """Scalar metrics only, suitable for JSON serialization."""
        return {
            "final_bankroll": self.final_bankroll,
            "total_profit": self.total_profit,
            "roi": self.roi,
            "accuracy": self.accuracy,
            "max_drawdown": self.max_drawdown,
            "n_bets": self.n_bets,
            "avg_edge": self.avg_edge,
            "clv": self.clv,
            "value_bets_rows": int(len(self.value_bets)),
        }

    def to_json(self, indent: int = 2) -> str:
        """JSON representation of scalar metrics only."""
        import json

        return json.dumps(self.to_dict(), indent=indent)


class Backtester:
    def __init__(self, config: BacktestConfig) -> None:
        self.cfg = config

    # --------------------------------------------------------
    # Core backtest
    # --------------------------------------------------------
    def run(self, df: pd.DataFrame) -> BacktestResult:
        """
        Run a backtest on a dataframe containing:
            - win_probability
            - closing_odds
            - odds
            - pick
            - actual_outcome
            - edge
        """
        logger.info(f"Starting backtest with {len(df)} rows")

        # Filter by min edge
        df = df[df["edge"] >= self.cfg.min_edge].copy()
        logger.info(f"{len(df)} bets after min-edge filter")

        if df.empty:
            return BacktestResult(
                final_bankroll=self.cfg.starting_bankroll,
                total_profit=0.0,
                roi=0.0,
                accuracy=0.0,
                max_drawdown=0.0,
                n_bets=0,
                avg_edge=0.0,
                clv=0.0,
                value_bets=pd.DataFrame(),
                bankroll_history=pd.DataFrame(),
                summary=pd.DataFrame(),
            )

        # ----------------------------------------------------
        # Kelly stake fraction
        # ----------------------------------------------------
        # Kelly fraction = (p*odds - 1) / (odds - 1)
        df["kelly_fraction_raw"] = df["edge"] / (df["odds"] - 1)

        # Apply fractional Kelly
        df["stake_fraction"] = df["kelly_fraction_raw"] * self.cfg.kelly_fraction

        # Cap stake fraction
        df["stake_fraction"] = df["stake_fraction"].clip(
            upper=self.cfg.max_stake_fraction
        )

        # ----------------------------------------------------
        # Bankroll simulation
        # ----------------------------------------------------
        bankroll = self.cfg.starting_bankroll
        bankroll_curve = [bankroll]

        stakes = []
        payouts = []

        for _, row in df.iterrows():
            stake = bankroll * row["stake_fraction"]
            stakes.append(stake)

            if row["actual_outcome"] == row["pick"]:
                payout = stake * (row["odds"] - 1)
            else:
                payout = -stake

            payouts.append(payout)

            bankroll += payout
            bankroll_curve.append(bankroll)

        df["stake"] = stakes
        df["payout"] = payouts

        bankroll_history = pd.DataFrame(
            {
                "bet_index": range(len(bankroll_curve)),
                "bankroll_after": bankroll_curve,
            }
        )

        # ----------------------------------------------------
        # CLV
        # ----------------------------------------------------
        df["implied_open"] = 1.0 / df["odds"]
        df["implied_close"] = 1.0 / df["closing_odds"]
        df["clv"] = df["implied_open"] - df["implied_close"]

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------
        final_bankroll = bankroll_curve[-1]
        total_profit = final_bankroll - self.cfg.starting_bankroll
        roi = total_profit / self.cfg.starting_bankroll
        accuracy = (df["actual_outcome"] == df["pick"]).mean()
        max_drawdown = self._compute_max_drawdown(bankroll_curve)
        avg_edge = df["edge"].mean()
        clv = df["clv"].mean()

        value_bets = df[df["edge"] >= self.cfg.min_edge].copy()

        return BacktestResult(
            final_bankroll=final_bankroll,
            total_profit=total_profit,
            roi=roi,
            accuracy=accuracy,
            max_drawdown=max_drawdown,
            n_bets=int(len(df)),
            avg_edge=avg_edge,
            clv=clv,
            value_bets=value_bets,
            bankroll_history=bankroll_history,
            summary=df,
        )

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------
    @staticmethod
    def _compute_max_drawdown(curve: list[float]) -> float:
        curve_arr = np.array(curve, dtype=float)
        peaks = np.maximum.accumulate(curve_arr)
        drawdowns = (curve_arr - peaks) / peaks
        return float(drawdowns.min()) if len(drawdowns) > 0 else 0.0
