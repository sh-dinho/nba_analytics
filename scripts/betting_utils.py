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
    Example: +150 -> 2.5, -150 -> 1.67
    """
    if american_odds == 0:
        raise DataError("American odds cannot be zero")

    if american_odds > 0:
        decimal_odds = 1 + (american_odds / 100)
    else:
        decimal_odds = 1 + (100 / abs(american_odds))

    result = round(decimal_odds, 2)
    logger.debug(f"Converted American odds {american_odds} → {result} (decimal)")
    return result


def expected_value(prob_win: float, american_odds: int, stake: float = 100.0) -> float:
    """
    Expected value of a bet given win probability, American odds, and stake size.
    Returns EV in dollars.
    """
    if not (0 <= prob_win <= 1):
        raise DataError("Probability must be between 0 and 1")
    if stake <= 0:
        raise DataError("Stake must be positive")

    decimal_odds = american_to_decimal(american_odds)
    profit_if_win = (decimal_odds - 1) * stake
    loss_if_lose = stake

    EV = (prob_win * profit_if_win) - ((1 - prob_win) * loss_if_lose)
    result = round(EV, 2)
    logger.debug(f"EV calc: prob_win={prob_win}, odds={american_odds}, stake={stake} → EV={result}")
    return result


def calculate_kelly_criterion(american_odds: int, model_prob: float,
                              bankroll: float = DEFAULT_BANKROLL,
                              max_fraction: float = MAX_KELLY_FRACTION) -> float:
    """
    Calculate Kelly bet size in dollars given odds, model probability, and bankroll.
    Caps bet size by max_fraction of bankroll.
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

    result = round(bet_size, 2)
    logger.debug(f"Kelly calc: prob={model_prob}, odds={american_odds}, bankroll={bankroll}, fraction={stake_fraction:.4f} → bet={result}")
    return result