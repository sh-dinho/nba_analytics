from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v5.0
# Name: Betting Mathematics
# File: src/app/engines/betting_math.py
# Purpose: Shared odds conversions, implied probability, and
#          EV-based edge utilities for all market engines.
# ============================================================

from typing import Optional


def american_to_decimal(odds: float) -> float:
    """Converts American odds to Decimal odds."""
    if odds == 0:
        raise ValueError("Odds cannot be zero.")
    if odds > 0:
        return 1 + (odds / 100.0)
    return 1 + (100.0 / abs(odds))


def decimal_to_american(decimal_odds: float) -> float:
    """Converts Decimal odds to American odds."""
    decimal_odds = max(decimal_odds, 1.0001)
    if decimal_odds >= 2.0:
        return (decimal_odds - 1.0) * 100.0
    return -100.0 / (decimal_odds - 1.0)


def implied_prob(odds: float) -> float:
    """Calculates implied probability from American odds."""
    if odds == 0:
        raise ValueError("Odds cannot be zero.")
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def calculate_edge(win_prob: float, odds: float) -> float:
    """
    Calculates EV-based edge: (Win Prob * Decimal Odds) - 1.
    win_prob may be in 0–1 or 0–100; normalized automatically.
    """
    if win_prob > 1:
        win_prob /= 100.0
    win_prob = min(max(win_prob, 0.0001), 0.9999)
    decimal_odds = american_to_decimal(odds)
    return (win_prob * decimal_odds) - 1
