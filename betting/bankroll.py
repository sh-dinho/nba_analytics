# betting/bankroll.py
import numpy as np

def simulate_bankroll(df, strategy="kelly", initial=1000.0, max_fraction=0.05):
    """
    df must have columns: decimal_odds, pred_home_win_prob
    Returns trajectory and metrics
    """
    bankroll = initial
    trajectory = []
    wins = 0
    losses = 0

    for i, row in df.iterrows():
        p = row["pred_home_win_prob"]
        o = row["decimal_odds"]
        b = o - 1

        if strategy == "kelly":
            fraction = max(0, min(((b*p)-(1-p))/b, max_fraction))
        else:
            fraction = max_fraction

        stake = bankroll * fraction
        won = np.random.rand() < p
        bankroll += stake*b if won else -stake
        bankroll = max(bankroll,0)
        trajectory.append(bankroll)
        wins += won
        losses += not won

    metrics = {
        "final_bankroll": bankroll,
        "roi": (bankroll - initial)/initial,
        "wins": wins,
        "losses": losses,
        "win_rate": wins/(wins+losses)
    }
    return trajectory, metrics
