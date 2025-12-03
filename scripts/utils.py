# ============================================================
# File: scripts/utils.py
# Purpose: Shared utility functions and bankroll simulation
# ============================================================

from typing import List, Dict
from core.log_config import setup_logger
from core.exceptions import DataError

logger = setup_logger("utils")


def expected_value(prob_win: float, odds: float, stake: float = 1.0) -> float:
    """
    Calculate expected value (EV) of a bet.
    EV = (prob_win * profit) - (prob_loss * stake)

    Args:
        prob_win (float): Probability of winning (0–1).
        odds (float): Decimal odds.
        stake (float): Stake size.

    Returns:
        float: Expected value of the bet.
    """
    if prob_win < 0 or prob_win > 1:
        raise DataError("Probability must be between 0 and 1")
    if odds <= 0:
        raise DataError("Odds must be positive")

    prob_loss = 1 - prob_win
    profit = (odds * stake) - stake
    ev = (prob_win * profit) - (prob_loss * stake)
    logger.debug(f"EV calculated: prob_win={prob_win}, odds={odds}, stake={stake}, EV={ev:.3f}")
    return ev


def kelly_fraction(prob_win: float, odds: float) -> float:
    """
    Calculate Kelly fraction for bet sizing.
    f* = (bp - q) / b

    Args:
        prob_win (float): Probability of winning (0–1).
        odds (float): Decimal odds.

    Returns:
        float: Kelly fraction (0–1).
    """
    if prob_win < 0 or prob_win > 1:
        raise DataError("Probability must be between 0 and 1")
    if odds <= 1:
        return 0

    b = odds - 1
    p = prob_win
    q = 1 - p
    kelly = (b * p - q) / b
    fraction = max(0, kelly)  # never negative
    logger.debug(f"Kelly fraction: prob_win={prob_win}, odds={odds}, fraction={fraction:.4f}")
    return fraction


def update_bankroll(bankroll: float, stake: float, won: bool, odds: float) -> float:
    """
    Update bankroll after a bet.

    Args:
        bankroll (float): Current bankroll.
        stake (float): Stake size.
        won (bool): Outcome of bet.
        odds (float): Decimal odds.

    Returns:
        float: Updated bankroll.
    """
    if won:
        profit = (odds * stake) - stake
        new_bankroll = bankroll + profit
    else:
        new_bankroll = bankroll - stake

    logger.debug(f"Bankroll updated: stake={stake}, won={won}, odds={odds}, new_bankroll={new_bankroll:.2f}")
    return new_bankroll


def implied_probability(odds: float) -> float:
    """
    Convert decimal odds to implied probability.

    Args:
        odds (float): Decimal odds.

    Returns:
        float: Implied probability.
    """
    if odds <= 0:
        raise DataError("Odds must be positive")
    prob = 1 / odds
    logger.debug(f"Implied probability: odds={odds}, prob={prob:.3f}")
    return prob


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
            fraction = min(kelly_fraction(prob_win, odds), max_fraction)
        else:  # flat betting
            fraction = max_fraction

        stake = self.bankroll * fraction
        ev = expected_value(prob_win, odds, stake)

        # If outcome not provided, simulate win if prob_win > 0.5
        won = outcome if outcome is not None else (prob_win > 0.5)
        self.bankroll = update_bankroll(self.bankroll, stake, won, odds)

        record = {
            "prob_win": prob_win,
            "odds": odds,
            "stake": stake,
            "won": won,
            "EV": ev,
            "bankroll": self.bankroll
        }
        self.history.append(record)
        logger.info(f"Bet placed: {record}")

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
        avg_ev = sum(h["EV"] for h in self.history) / total_bets if total_bets > 0 else 0
        avg_stake = sum(h["stake"] for h in self.history) / total_bets if total_bets > 0 else 0

        summary = {
            "Final_Bankroll": self.bankroll,
            "Total_Bets": total_bets,
            "Win_Rate": win_rate,
            "Avg_EV": avg_ev,
            "Avg_Stake": avg_stake
        }
        logger.info(f"Simulation summary: {summary}")
        return summary