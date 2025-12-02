import pandas as pd
import numpy as np

def simulate_bankroll(df, strategy="kelly", max_fraction=0.05, initial=1000.0):
    """
    Simulate bankroll based on predictions and odds.
    df must contain: decimal_odds, prob, ev
    """
    bankroll = initial
    trajectory = []
    wins = 0
    losses = 0

    for i, row in df.iterrows():
        p = row["prob"]
        o = row["decimal_odds"]
        b = o - 1

        if strategy=="kelly":
            fraction = max(0, min((b*p-(1-p))/b if b>0 else 0, max_fraction))
        else:
            fraction = max_fraction

        stake = bankroll * fraction
        won = np.random.rand() < p
        bankroll += stake*b if won else -stake
        bankroll = max(bankroll, 0)
        trajectory.append(bankroll)
        wins += won
        losses += not won

    metrics = {
        "final_bankroll_mean": bankroll,
        "roi": (bankroll-initial)/initial,
        "wins": wins,
        "losses": losses,
        "win_rate": wins/(wins+losses)
    }

    return trajectory, metrics
