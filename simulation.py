import numpy as np
from typing import List, Tuple

def simulate_bankroll(
    probs: List[float],
    odds: float,
    starting_bankroll: float,
    method: str,
    confidence_factor: float,
    min_stake: float,
    games: int,
    transaction_fee: float = 0.0,
    seed: int = None
) -> Tuple[List[float], float, float, float, float]:
    """
    Simulate bankroll progression over a series of bets.
    """
    if starting_bankroll <= 0:
        raise ValueError("Starting bankroll must be positive.")
    if odds <= 1.0:
        raise ValueError("Odds must be greater than 1.0.")
    if games <= 0:
        raise ValueError("Number of games must be positive.")

    rng = np.random.default_rng(seed)
    probs = np.asarray(probs).ravel()
    probs = probs[:games] if len(probs) >= games else np.pad(probs, (0, games - len(probs)), constant_values=0.5)

    bankroll = starting_bankroll
    history = [bankroll]
    wins = 0
    peak = bankroll
    max_drawdown = 0.0

    for p in probs:
        if method.lower() == "kelly":
            kelly = max(0.0, (p * (odds - 1) - (1 - p)) / (odds - 1))
            stake = max(min_stake, bankroll * kelly * confidence_factor)
        else:
            stake = max(min_stake, bankroll * confidence_factor)

        outcome = rng.random() < p
        if outcome:
            bankroll += stake * (odds - 1) * (1 - transaction_fee)
            wins += 1
        else:
            bankroll -= stake

        peak = max(peak, bankroll)
        drawdown = (peak - bankroll) / peak if peak > 0 else 0
        max_drawdown = max(max_drawdown, drawdown)
        history.append(bankroll)

    final_bankroll = bankroll
    roi = (final_bankroll - starting_bankroll) / starting_bankroll
    win_rate = wins / games if games > 0 else 0.0

    return history, final_bankroll, roi, win_rate, max_drawdown