# File: scripts/simulate_bankroll.py

import pandas as pd

def simulate_bankroll(df: pd.DataFrame, strategy: str = "kelly", initial: float = 1000):
    """
    df columns: decimal_odds, prob, ev_home (ev used for screening/sizing if needed)
    Returns: (trajectory_list, metrics_dict)
    """
    bankroll = initial
    trajectory = []
    wins = losses = 0

    for _, row in df.iterrows():
        p = float(row["prob"])
        o = float(row["decimal_odds"])
        # Basic Kelly fraction for decimal odds (assuming net odds b = o-1)
        b = o - 1.0
        kelly = ((b * p) - (1 - p)) / b if b > 0 else 0
        fraction = max(0.0, min(kelly, 0.05))  # cap at 5%
        stake = bankroll * fraction

        # Simulate expected outcome deterministically? If you want stochastic, replace with random:
        expected_edge = (p * b) - (1 - p)
        won = expected_edge > 0  # deterministic proxy; replace with random draw for Monte Carlo
        if won:
            bankroll += stake * b
            wins += 1
        else:
            bankroll -= stake
            losses += 1

        trajectory.append(bankroll)

    roi = (bankroll - initial) / initial if initial else 0.0
    metrics = {
        "final_bankroll": bankroll,
        "roi": roi,
        "wins": wins,
        "losses": losses,
        "win_rate": wins / (wins + losses) if (wins + losses) else 0.0,
    }
    return trajectory, metrics