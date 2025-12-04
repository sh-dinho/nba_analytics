# ============================================================
# File: scripts/betting_utils.py
# Purpose: Odds conversion, expected value, and Kelly criterion with custom stake support
# ============================================================

from typing import Tuple, Union
from core.log_config import init_global_logger
from core.exceptions import DataError
from core.config import DEFAULT_BANKROLL, MAX_KELLY_FRACTION

logger = init_global_logger()


def american_to_decimal(american_odds: Union[int, float]) -> float:
    """
    Convert American odds to decimal odds (European odds).

    Formula:
    - Positive odds: decimal = 1 + (odds / 100)
    - Negative odds: decimal = 1 + (100 / abs(odds))

    Examples:
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
    return float(decimal_odds)


def _validate_prob(p: float) -> float:
    """Validate and clip probability to (0,1) for numerical stability."""
    if not (0.0 <= p <= 1.0):
        raise DataError("Probability must be between 0 and 1")
    eps = 1e-6
    return max(min(p, 1.0 - eps), eps)


def _validate_decimal_odds(decimal_odds: float) -> float:
    """Validate decimal odds; must be >= 1.01."""
    if decimal_odds is None:
        raise DataError("Decimal odds cannot be None")
    if decimal_odds < 1.01:
        raise DataError(f"Decimal odds must be >= 1.01, got {decimal_odds}")
    return float(decimal_odds)


def _profit_if_win(decimal_odds: float, stake: float) -> float:
    """Profit if bet wins given decimal odds and stake."""
    return (decimal_odds - 1.0) * stake


def expected_value_decimal(prob_win: float, decimal_odds: float, stake: float = 100.0) -> float:
    """
    Calculate expected value (EV) using decimal odds.

    EV = (p * profit_if_win) - ((1 - p) * stake)
    """
    p = _validate_prob(prob_win)
    d = _validate_decimal_odds(decimal_odds)
    if stake <= 0:
        raise DataError("Stake must be positive")

    profit_if_win = _profit_if_win(d, stake)
    ev = (p * profit_if_win) - ((1.0 - p) * stake)
    logger.debug(f"EV (decimal): p={p:.6f}, d_odds={d:.3f}, stake={stake:.2f} → EV={ev:.2f}")
    return float(ev)


def expected_value(prob_win: float, american_odds: Union[int, float], stake: float = 100.0) -> float:
    """Calculate expected value (EV) using American odds."""
    decimal_odds = american_to_decimal(american_odds)
    return expected_value_decimal(prob_win, decimal_odds, stake)


def kelly_fraction_decimal(prob_win: float, decimal_odds: float) -> float:
    """
    Calculate Kelly fraction given probability and decimal odds.

    Kelly fraction = (bp - q) / b
    where:
        b = decimal_odds - 1
        p = prob_win
        q = 1 - p
    """
    p = _validate_prob(prob_win)
    d = _validate_decimal_odds(decimal_odds)
    b = d - 1.0
    if b <= 0:
        return 0.0
    q = 1.0 - p
    f = (b * p - q) / b
    return float(max(f, 0.0))


def kelly_fraction(prob_win: float, american_odds: Union[int, float]) -> float:
    """Kelly fraction using American odds."""
    decimal_odds = american_to_decimal(american_odds)
    return kelly_fraction_decimal(prob_win, decimal_odds)


def calculate_kelly_criterion_decimal(
    decimal_odds: float,
    model_prob: float,
    bankroll: float = DEFAULT_BANKROLL,
    max_fraction: float = MAX_KELLY_FRACTION,
) -> float:
    """
    Calculate Kelly bet size in dollars given decimal odds, model probability, and bankroll.
    Caps by max_fraction.
    """
    if bankroll <= 0:
        raise DataError("Bankroll must be positive")

    fraction = kelly_fraction_decimal(model_prob, decimal_odds)
    stake_fraction = min(max(fraction, 0.0), float(max_fraction))
    bet_size = bankroll * stake_fraction

    logger.debug(
        f"Kelly (decimal): p={model_prob:.6f}, d_odds={decimal_odds:.3f}, "
        f"bankroll={bankroll:.2f}, fraction={stake_fraction:.4f} → bet={bet_size:.2f}"
    )
    return float(bet_size)


def calculate_kelly_criterion(
    american_odds: Union[int, float],
    model_prob: float,
    bankroll: float = DEFAULT_BANKROLL,
    max_fraction: float = MAX_KELLY_FRACTION,
) -> float:
    """Kelly stake using American odds."""
    decimal_odds = american_to_decimal(american_odds)
    return calculate_kelly_criterion_decimal(decimal_odds, model_prob, bankroll, max_fraction)


def ev_and_kelly_summary_decimal(
    prob_win: float,
    decimal_odds: float,
    bankroll: float = DEFAULT_BANKROLL,
    stake: float = 100.0,
    max_fraction: float = MAX_KELLY_FRACTION,
) -> Tuple[float, float]:
    """Convenience helper returning (EV, Kelly bet size) using decimal odds."""
    ev = expected_value_decimal(prob_win, decimal_odds, stake)
    kelly_bet = calculate_kelly_criterion_decimal(decimal_odds, prob_win, bankroll, max_fraction)
    return float(ev), float(kelly_bet)


def ev_and_kelly_summary(
    prob_win: float,
    american_odds: Union[int, float],
    bankroll: float = DEFAULT_BANKROLL,
    stake: float = 100.0,
    max_fraction: float = MAX_KELLY_FRACTION,
) -> Tuple[float, float]:
    """Convenience helper returning (EV, Kelly bet size) using American odds."""
    decimal_odds = american_to_decimal(american_odds)
    ev = expected_value_decimal(prob_win, decimal_odds, stake)
    kelly_bet = calculate_kelly_criterion_decimal(decimal_odds, prob_win, bankroll, max_fraction)
    return float(ev), float(kelly_bet)
