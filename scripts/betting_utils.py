# ============================================================
# File: scripts/betting_utils.py
# Purpose: Odds conversion, expected value, and Kelly criterion with custom stake support
# ============================================================

from core.log_config import setup_logger
from core.exceptions import DataError
from core.config import DEFAULT_BANKROLL, MAX_KELLY_FRACTION

logger = setup_logger("betting_utils")


def american_to_decimal(american_odds: int) -> float:
    """
    Convert American odds to decimal odds (European odds).
    
    Formula:
    - Positive odds: decimal = 1 + (odds / 100)
    - Negative odds: decimal = 1 + (100 / abs(odds))

    Example:
        +150 -> 2.5
        -150 -> 1.67
    """
    if american_odds == 0:
        raise DataError("American odds cannot be zero")

    if american_odds > 0:
        decimal_odds = 1 + (american_odds / 100)
    else:
        decimal_odds = 1 + (100 / abs(american_odds))

    logger.debug(f"Converted American odds {american_odds} → {decimal_odds:.4f} (decimal)")
    return decimal_odds


def expected_value(prob_win: float, american_odds: int, stake: float = 100.0) -> float:
    """
    Calculate expected value (EV) of a bet.

    EV = (p * profit_if_win) - (q * loss_if_lose)

    Args:
        prob_win: Probability of winning (0 ≤ p ≤ 1).
        american_odds: American odds (e.g., +150, -110).
        stake: Stake size in dollars.

    Returns:
        EV in dollars (float).
    """
    if not (0 <= prob_win <= 1):
        raise DataError("Probability must be between 0 and 1")
    if stake <= 0:
        raise DataError("Stake must be positive")

    decimal_odds = american_to_decimal(american_odds)
    profit_if_win = (decimal_odds - 1) * stake
    loss_if_lose = stake

    ev = (prob_win * profit_if_win) - ((1 - prob_win) * loss_if_lose)
    logger.debug(f"EV calc: prob_win={prob_win:.3f}, odds={american_odds}, stake={stake:.2f} → EV={ev:.2f}")
    return ev


def calculate_kelly_criterion(
    american_odds: int,
    model_prob: float,
    bankroll: float = DEFAULT_BANKROLL,
    max_fraction: float = MAX_KELLY_FRACTION
) -> float:
    """
    Calculate Kelly bet size in dollars given odds, model probability, and bankroll.

    Kelly fraction = (bp - q) / b
    where:
        b = decimal_odds - 1
        p = model probability
        q = 1 - p

    Args:
        american_odds: American odds (e.g., +150, -110).
        model_prob: Model-estimated win probability (0 ≤ p ≤ 1).
        bankroll: Current bankroll in dollars.
        max_fraction: Maximum fraction of bankroll to risk.

    Returns:
        Bet size in dollars (float).
    """
    if not (0 <= model_prob <= 1):
        raise DataError("Model probability must be between 0 and 1")
    if bankroll <= 0:
        raise DataError("Bankroll must be positive")

    decimal_odds = american_to_decimal(american_odds)
    b = decimal_odds - 1
    if b <= 0:
        return 0.0

    p = model_prob
    q = 1 - p
    kelly_fraction = (b * p - q) / b
    kelly_fraction = max(kelly_fraction, 0.0)

    # Cap by max_fraction
    stake_fraction = min(kelly_fraction, max_fraction)
    bet_size = bankroll * stake_fraction

    logger.debug(
        f"Kelly calc: prob={model_prob:.3f}, odds={american_odds}, "
        f"bankroll={bankroll:.2f}, fraction={stake_fraction:.4f} → bet={bet_size:.2f}"
    )
    return bet_size