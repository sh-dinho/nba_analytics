# ============================================================
# File: pipelines/strategy.py
# Purpose: Apply bankroll management strategies (Kelly or Flat) with simulation support
# ============================================================

import numpy as np
import pandas as pd
from typing import List, Dict
from core.log_config import setup_logger
from core.exceptions import PipelineError, DataError
from core.config import DEFAULT_BANKROLL, MAX_KELLY_FRACTION

logger = setup_logger("strategy")

# --- Utility Functions ---

def Expected_Value(prob_win: float, odds: float, stake: float = 1.0) -> float:
    """Calculate expected value (EV) of a bet."""
    if not (0 <= prob_win <= 1):
        raise DataError("Probability must be between 0 and 1")
    if odds <= 0:
        raise DataError("Odds must be positive")

    prob_loss = 1 - prob_win
    profit = (odds * stake) - stake
    return (prob_win * profit) - (prob_loss * stake)


def Kelly_Fraction(prob_win: float, odds: float) -> float:
    """Calculate Kelly fraction for bet sizing."""
    if not (0 <= prob_win <= 1):
        raise DataError("Probability must be between 0 and 1")
    b = odds - 1
    if b <= 0:
        return 0
    p = prob_win
    q = 1 - p
    kelly = (b * p - q) / b
    return max(0, kelly)


def Update_Bankroll(bankroll: float, stake: float, won: bool, odds: float) -> float:
    """Update bankroll after a bet outcome."""
    if won:
        profit = (odds * stake) - stake
        return bankroll + profit
    else:
        return bankroll - stake


def Implied_Probability(odds: float) -> float:
    """Convert decimal odds to implied probability."""
    return 1 / odds if odds > 0 else 0


# --- Simulation Class ---

class Simulation:
    """Run bankroll simulations across multiple bets."""

    def __init__(self, initial_bankroll: float = DEFAULT_BANKROLL):
        self.bankroll = initial_bankroll
        self.history: List[Dict] = []
        self.trajectory: List[float] = [initial_bankroll]

    def place_bet(self, prob_win: float, odds: float,
                  strategy: str = "kelly", max_fraction: float = MAX_KELLY_FRACTION,
                  outcome: bool = None):
        """Place a bet using a given strategy."""
        if strategy == "kelly":
            fraction = min(Kelly_Fraction(prob_win, odds), max_fraction)
        elif strategy == "flat":
            fraction = max_fraction
        else:
            raise PipelineError(f"Unknown strategy: {strategy}")

        stake = self.bankroll * fraction
        ev = Expected_Value(prob_win, odds, stake)

        # If outcome not provided, simulate stochastically
        won = outcome if outcome is not None else (np.random.rand() < prob_win)
        self.bankroll = Update_Bankroll(self.bankroll, stake, won, odds)
        self.trajectory.append(self.bankroll)

        self.history.append({
            "prob_win": prob_win,
            "odds": odds,
            "stake": stake,
            "won": won,
            "EV": ev,
            "bankroll": self.bankroll
        })

    def run(self, bets: List[Dict], strategy: str = "kelly", max_fraction: float = MAX_KELLY_FRACTION):
        """Run simulation across a list of bets."""
        for bet in bets:
            self.place_bet(
                prob_win=bet["prob_win"],
                odds=bet["odds"],
                strategy=strategy,
                max_fraction=max_fraction,
                outcome=bet.get("won")
            )
        return self.history

    def summary(self) -> Dict:
        """Return final bankroll and stats."""
        total_bets = len(self.history)
        wins = sum(1 for h in self.history if h["won"])
        win_rate = wins / total_bets if total_bets > 0 else 0
        return {
            "Final_Bankroll": self.bankroll,
            "Total_Bets": total_bets,
            "Win_Rate": win_rate,
            "Avg_EV": sum(h["EV"] for h in self.history) / total_bets if total_bets > 0 else 0,
            "Avg_Stake": sum(h["stake"] for h in self.history) / total_bets if total_bets > 0 else 0
        }


# --- Pipeline Integration Function ---

def apply_strategy(predictions_df: pd.DataFrame,
                   threshold: float = 0.6,
                   strategy: str = "kelly",
                   max_fraction: float = MAX_KELLY_FRACTION,
                   initial_bankroll: float = DEFAULT_BANKROLL):
    """
    Apply bankroll management strategy to predictions DataFrame.
    Returns picks_df, metrics, trajectory.
    """
    if "pred_home_win_prob" not in predictions_df.columns:
        raise DataError("Predictions DataFrame missing 'pred_home_win_prob' column")

    sim = Simulation(initial_bankroll=initial_bankroll)

    bets = []
    for _, row in predictions_df.iterrows():
        prob = row["pred_home_win_prob"]
        if prob < threshold:
            continue
        bets.append({"prob_win": prob, "odds": 2.0})  # assume even odds for now

    sim.run(bets, strategy=strategy, max_fraction=max_fraction)
    metrics = sim.summary()

    picks_df = pd.DataFrame(sim.history)
    picks_df["bankroll_after"] = [h["bankroll"] for h in sim.history]

    logger.info(f"Strategy applied: {strategy}, Final bankroll={metrics['Final_Bankroll']:.2f}, ROI={(metrics['Final_Bankroll']-initial_bankroll)/initial_bankroll:.2%}")
    return picks_df, metrics, sim.trajectory