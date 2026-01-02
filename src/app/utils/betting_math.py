from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v5.0
# Module: Betting Mathematics (Canonical)
# Purpose:
#     Unified, type‑safe, market‑agnostic math utilities:
#       • American ↔ Decimal conversions
#       • Implied probability
#       • No‑vig probability
#       • EV / Edge
#       • Parlay math
#       • Kelly sizing
# ============================================================

from typing import Tuple


# ------------------------------------------------------------
# Odds Conversions
# ------------------------------------------------------------
def american_to_decimal(odds: float) -> float:
    """
    Convert American odds to decimal odds.

    Examples:
        +150 → 2.50
        -120 → 1.83
    """
    if odds > 0:
        return 1 + (odds / 100.0)
    return 1 + (100.0 / abs(odds))


def decimal_to_american(decimal_odds: float) -> float:
    """
    Convert decimal odds to American odds.

    Examples:
        2.50 → +150
        1.83 → -120
    """
    if decimal_odds >= 2.0:
        return (decimal_odds - 1.0) * 100.0
    return -100.0 / (decimal_odds - 1.0)


# ------------------------------------------------------------
# Implied Probability
# ------------------------------------------------------------
def implied_prob(odds: float) -> float:
    """
    Convert American odds to implied probability.

    Examples:
        +150 → 0.40
        -120 → 0.545
    """
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


# ------------------------------------------------------------
# No‑Vig Probability (Fair Probability)
# ------------------------------------------------------------
def remove_vig(odds_a: float, odds_b: float) -> Tuple[float, float]:
    """
    Convert two-sided market odds into fair (no-vig) probabilities.

    Args:
        odds_a: American odds for side A
        odds_b: American odds for side B

    Returns:
        (pA_fair, pB_fair)
    """
    pA = implied_prob(odds_a)
    pB = implied_prob(odds_b)

    total = pA + pB
    if total == 0:
        return 0.0, 0.0

    return pA / total, pB / total


# ------------------------------------------------------------
# Expected Value / Edge
# ------------------------------------------------------------
def calculate_edge(win_prob: float, odds: float) -> float:
    """
    Calculate EV-based edge:
        Edge = (win_prob * decimal_odds) - 1

    Returns:
        Positive → profitable long-term
        Negative → losing long-term
    """
    decimal_odds = american_to_decimal(odds)
    return (win_prob * decimal_odds) - 1.0


def expected_value(win_prob: float, odds: float, stake: float = 1.0) -> float:
    """
    Expected value in dollars for a given stake.

    EV = (win_prob * profit) - (loss_prob * stake)
    """
    dec = american_to_decimal(odds)
    profit = (dec - 1.0) * stake
    loss_prob = 1.0 - win_prob
    return (win_prob * profit) - (loss_prob * stake)


# ------------------------------------------------------------
# Parlay Math
# ------------------------------------------------------------
def parlay_decimal_odds(legs: list[float]) -> float:
    """
    Combine decimal odds for a parlay.

    Args:
        legs: list of decimal odds for each leg
    """
    total = 1.0
    for d in legs:
        total *= d
    return total


def parlay_win_prob(legs: list[float]) -> float:
    """
    Combine win probabilities for independent parlay legs.
    """
    p = 1.0
    for w in legs:
        p *= w
    return p


# ------------------------------------------------------------
# Kelly Criterion
# ------------------------------------------------------------
def kelly_fraction(win_prob: float, odds: float) -> float:
    """
    Kelly fraction for optimal bet sizing.

    Args:
        win_prob: model-estimated win probability
        odds: American odds

    Returns:
        Fraction of bankroll to wager (0–1).
    """
    dec = american_to_decimal(odds)
    b = dec - 1.0  # net profit multiplier

    return max(0.0, (win_prob * (b + 1) - 1) / b)