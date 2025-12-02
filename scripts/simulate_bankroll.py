# scripts/simulate_bankroll.py
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any
from scripts.utils import setup_logger

logger = setup_logger("simulate_bankroll")

def simulate_bankroll(
    df: pd.DataFrame,
    strategy: str = "kelly",
    initial: float = 1000.0,
    max_fraction: float = 0.05
) -> Tuple[List[float], Dict[str, Any]]:
    """
    Simulate bankroll based on predictions and strategy.
    """
    bankroll = initial
    history = []

    probs = df["pred_home_win_prob"].values
    odds = df["decimal_odds"].values

    wins = 0
    losses = 0

    for i in range(len(df)):
        p = probs[i]
        o = odds[i]
        b = o - 1

        if strategy == "kelly":
            fraction = max(0, min(((b*p)-(1-p))/b if b>0 else 0, max_fraction))
        else:
            fraction = max_fraction

        stake = bankroll * fraction
        won = np.random.rand() < p
        bankroll += stake*b if won else -stake
        bankroll = max(bankroll, 0)
        history.append(bankroll)

        if won:
            wins += 1
        else:
            losses += 1

    metrics = {
        "final_bankroll": bankroll,
        "roi": (bankroll - initial)/initial,
        "wins": wins,
        "losses": losses,
        "win_rate": wins/(wins+losses) if (wins+losses)>0 else 0
    }

    logger.info(f"Simulation complete: Final bankroll ${bankroll:.2f}")
    return history, metrics
