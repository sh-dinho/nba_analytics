# File: betting/utils.py
# Unified utilities for odds conversion, expected value, and Kelly fraction calculation.

def american_to_decimal(american_odds: int) -> float:
    """Converts American odds to decimal odds (European odds)."""
    if american_odds > 0:
        decimal_odds = 1 + (american_odds / 100)
    else:
        decimal_odds = 1 + (100 / abs(american_odds))
    return round(decimal_odds, 3)

def expected_value(Pwin: float, decimal_odds: float) -> float:
    """Calculates the Expected Value (EV) of a bet for a unit stake."""
    profit_if_win = decimal_odds - 1
    Ploss = 1 - Pwin
    EV = (Pwin * profit_if_win) - Ploss
    return round(EV, 5)

def calculate_kelly_fraction(Pwin: float, decimal_odds: float, max_fraction: float) -> float:
    """Calculates the Kelly Criterion fraction of bankroll to bet."""
    p = Pwin
    q = 1 - p
    b = decimal_odds - 1

    if b <= 0.001: # Avoid division by zero or negative profit ratios
        return 0.0

    kelly_fraction = ((b * p) - q) / b
    
    # Clamp the fraction: min is 0, max is the MAX_FRACTION limit
    return max(0.0, min(kelly_fraction, max_fraction))