# ============================================================
# File: scripts/Utils.py
# Purpose: Shared utility functions and bankroll simulation
# ============================================================

from typing import List, Dict


def Expected_Value(prob_win: float, odds: float, stake: float = 1.0) -> float:
    """
    Calculate expected value (EV) of a bet.
    EV = (prob_win * profit) - (prob_loss * stake)
    """
    prob_loss = 1 - prob_win
    profit = (odds * stake) - stake
    return (prob_win * profit) - (prob_loss * stake)


def Kelly_Fraction(prob_win: float, odds: float) -> float:
    """
    Calculate Kelly fraction for bet sizing.
    f* = (bp - q) / b
    """
    b = odds - 1
    p = prob_win
    q = 1 - p
    kelly = (b * p - q) / b if b > 0 else 0
    return max(0, kelly)  # never negative


def Update_Bankroll(bankroll: float, stake: float, won: bool, odds: float) -> float:
    """
    Update bankroll after a bet.
    """
    if won:
        profit = (odds * stake) - stake
        return bankroll + profit
    else:
        return bankroll - stake


def Implied_Probability(odds: float) -> float:
    """
    Convert decimal odds to implied probability.
    """
    return 1 / odds if odds > 0 else 0


class Simulation:
    """
    Run bankroll simulations across multiple bets.
    """

    def __init__(self, initial_bankroll: float = 1000.0):
        self.bankroll = initial_bankroll
        self.history: List[Dict] = []

    def place_bet(self, prob_win: float, odds: float,
                  strategy: str = "kelly", max_fraction: float = 0.05,
                  outcome: bool = None):
        """
        Place a bet using a given strategy.
        Args:
            prob_win (float): Probability of winning (0–1).
            odds (float): Decimal odds.
            strategy (str): "kelly" or "flat".
            max_fraction (float): Max fraction of bankroll to wager.
            outcome (bool): Optional explicit outcome (True=win, False=loss).
        """
        if strategy == "kelly":
            fraction = min(Kelly_Fraction(prob_win, odds), max_fraction)
        else:  # flat betting
            fraction = max_fraction

        stake = self.bankroll * fraction
        ev = Expected_Value(prob_win, odds, stake)

        # If outcome not provided, simulate win if prob_win > 0.5
        won = outcome if outcome is not None else (prob_win > 0.5)
        self.bankroll = Update_Bankroll(self.bankroll, stake, won, odds)

        self.history.append({
            "prob_win": prob_win,
            "odds": odds,
            "stake": stake,
            "won": won,
            "EV": ev,
            "bankroll": self.bankroll
        })

    def run(self, bets: List[Dict], strategy: str = "kelly", max_fraction: float = 0.05):
        """
        Run simulation across a list of bets.
        Args:
            bets (list): List of dicts with {"prob_win": float, "odds": float}.
        """
        for bet in bets:
            self.place_bet(
                prob_win=bet["prob_win"],
                odds=bet["odds"],
                strategy=strategy,
                max_fraction=max_fraction,
                outcome=bet.get("won")  # allow explicit outcomes
            )
        return self.history

    def summary(self) -> Dict:
        """
        Return final bankroll and stats.
        """
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