# ============================================================
# File: scripts/betting_utils.py
# Purpose: Odds conversion, expected value, and Kelly criterion with custom stake support
# ============================================================

def american_to_decimal(american_odds: int) -> float:
    """
    Converts American odds to decimal odds (European odds).
    """
    if american_odds > 0:
        # Positive odds: +150 -> 2.5
        decimal_odds = 1 + (american_odds / 100)
    else:
        # Negative odds: -150 -> 1.67
        decimal_odds = 1 + (100 / abs(american_odds))
    return round(decimal_odds, 2)


def expected_value(Pwin: float, american_odds: int, stake: float = 100.0) -> float:
    """
    Expected value of a bet given win probability, American odds, and stake size.
    Returns EV in dollars.
    """
    decimal_odds = american_to_decimal(american_odds)
    profit_if_win = (decimal_odds - 1) * stake
    loss_if_lose = stake

    EV = (Pwin * profit_if_win) - ((1 - Pwin) * loss_if_lose)
    return round(EV, 2)


def calculate_kelly_criterion(american_odds: int, model_prob: float, bankroll: float, stake_unit: float = 100.0) -> float:
    """
    Calculates the Kelly bet size in dollars given odds, model probability, and bankroll.
    """
    decimal_odds = american_to_decimal(american_odds)
    b = decimal_odds - 1
    p = model_prob
    q = 1 - p

    kelly_fraction = (b * p - q) / b
    kelly_fraction = max(kelly_fraction, 0.0)  # no negative bets

    # Bet size in dollars
    bet_size = bankroll * kelly_fraction
    return round(bet_size, 2)