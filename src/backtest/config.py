from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Backtest Configuration
# File: src/backtest/config.py
# Author: Sadiq
#
# Description:
#     Configuration object for bankroll management and bet sizing
#     during backtests. Version‑agnostic and aligned with the
#     modern modeling architecture.
# ============================================================

from dataclasses import dataclass


@dataclass
class BacktestConfig:
    """
    Configuration for the backtester.

    Parameters
    ----------
    starting_bankroll : float
        Initial bankroll at the start of the simulation.
    min_edge : float
        Minimum required edge (expected value advantage) to place a bet.
    kelly_fraction : float
        Fraction of the Kelly stake to wager.
    max_stake_fraction : float
        Maximum fraction of bankroll allowed on any single bet.
    """

    starting_bankroll: float = 1000.0
    min_edge: float = 0.03
    kelly_fraction: float = 0.25
    max_stake_fraction: float = 0.05

    def __post_init__(self):
        if not (0 <= self.kelly_fraction <= 1):
            raise ValueError("kelly_fraction must be between 0 and 1")

        if not (0 <= self.max_stake_fraction <= 1):
            raise ValueError("max_stake_fraction must be between 0 and 1")

        if self.min_edge < 0:
            raise ValueError("min_edge must be non‑negative")

    def compute_stake(self, bankroll: float, edge: float) -> float:
        """
        Compute the stake size using fractional Kelly with a cap.

        Parameters
        ----------
        bankroll : float
            Current bankroll.
        edge : float
            Expected value advantage (p*odds - 1).

        Returns
        -------
        float
            Stake amount.
        """
        if edge < self.min_edge:
            return 0.0

        kelly_stake = bankroll * edge * self.kelly_fraction
        max_stake = bankroll * self.max_stake_fraction

        return min(kelly_stake, max_stake)