# ============================================================
# 🏀 NBA Analytics v4
# Module: Betting Mathematics
# Author: Sadiq
# Version: 4.0.0
# Purpose: Unified conversion and EV logic for all market engines.
# ============================================================
from __future__ import annotations
from typing import Optional


def american_to_decimal(odds: float) -> float:
    """Converts American odds to Decimal odds."""
    if odds > 0:
        return 1 + (odds / 100.0)
    return 1 + (100.0 / abs(odds))


def decimal_to_american(decimal_odds: float) -> float:
    """Converts Decimal odds to American odds."""
    if decimal_odds >= 2.0:
        return (decimal_odds - 1.0) * 100.0
    return -100.0 / (decimal_odds - 1.0)


def implied_prob(odds: float) -> float:
    """Calculates implied probability from American odds."""
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def calculate_edge(win_prob: float, odds: float) -> float:
    """Calculates EV-based edge: (Win Prob * Decimal Odds) - 1."""
    decimal_odds = american_to_decimal(odds)
    return (win_prob * decimal_odds) - 1
