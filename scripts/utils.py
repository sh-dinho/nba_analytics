# ============================================================
# File: scripts/utils.py
# Purpose: Shared utility functions and bankroll simulation
# ============================================================

from typing import List, Dict
from core.log_config import setup_logger
from core.exceptions import DataError
import os
import datetime
import pandas as pd

logger = setup_logger("utils")

# -----------------------------
# General Helpers
# -----------------------------

def get_timestamp() -> str:
    """Return current timestamp string for filenames/logs."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_columns(df: pd.DataFrame, required_cols: list, label: str = "dataframe") -> pd.DataFrame:
    """
    Ensure DataFrame has required columns, add if missing.

    Parameters
    ----------
    df : pandas.DataFrame
    required_cols : list
        Required column names.
    label : str, optional
        Label for error messages.

    Returns
    -------
    pandas.DataFrame
        DataFrame with missing columns added as None.
    """
    for col in required_cols:
        if col not in df.columns:
            df[col] = None
            logger.warning(f"⚠️ Added missing column '{col}' to {label}")
    return df

def append_pipeline_summary(summary_file: str, metrics: dict):
    """Append metrics dict to pipeline summary CSV."""
    row = pd.DataFrame([metrics])
    if os.path.exists(summary_file):
        df = pd.read_csv(summary_file)
        df = pd.concat([df, row], ignore_index=True)
    else:
        df = row
    df.to_csv(summary_file, index=False)
    logger.info(f"📊 Pipeline summary updated at {summary_file}")

# -----------------------------
# Betting Math
# -----------------------------

def expected_value(prob_win: float, odds: float, stake: float = 1.0) -> float:
    """Calculate expected value (EV) of a bet."""
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
    """Calculate Kelly fraction for bet sizing."""
    if prob_win < 0 or prob_win > 1:
        raise DataError("Probability must be between 0 and 1")
    if odds <= 1:
        return 0

    b = odds - 1
    p = prob_win
    q = 1 - p
    kelly = (b * p - q) / b
    fraction = max(0, kelly)
    logger.debug(f"Kelly fraction: prob_win={prob_win}, odds={odds}, fraction={fraction:.4f}")
    return fraction

def update_bankroll(bankroll: float, stake: float, won: bool, odds: float) -> float:
    """Update bankroll after a bet."""
    if won:
        profit = (odds * stake) - stake
        new_bankroll = bankroll + profit
    else:
        new_bankroll = bankroll - stake
    logger.debug(f"Bankroll updated: stake={stake}, won={won}, odds={odds}, new_bankroll={new_bankroll:.2f}")
    return new_bankroll

def implied_probability(odds: float) -> float:
    """Convert decimal odds to implied probability."""
    if odds <= 0:
        raise DataError("Odds must be positive")
    prob = 1 / odds
    logger.debug(f"Implied probability: odds={odds}, prob={prob:.3f}")
    return prob

# -----------------------------
# EV Highlight Helper
# -----------------------------

def log_positive_ev(df: pd.DataFrame, ev_col: str = "ev", prob_col: str = "pred_home_win_prob",
                    odds_col: str = "decimal_odds", home_col: str = "home_team", away_col: str = "away_team"):
    """
    Log all rows with positive expected value (EV > 0).

    Parameters
    ----------
    df : pandas.DataFrame
        Predictions DataFrame with EV column.
    ev_col : str
        Column name for expected value.
    """
    if ev_col not in df.columns:
        logger.warning("⚠️ No EV column found in predictions DataFrame")
        return

    positive_ev = df[df[ev_col] > 0]
    if positive_ev.empty:
        logger.info("No positive EV picks found.")
    else:
        logger.info("=== POSITIVE EV PICKS ===")
        for _, row in positive_ev.iterrows():
            logger.info(
                f"{row.get(home_col, 'Home')} vs {row.get(away_col, 'Away')} → "
                f"Prob={row.get(prob_col, 0):.3f}, Odds={row.get(odds_col, 0)}, EV={row[ev_col]:.3f}"
            )

# -----------------------------
# Simulation Class
# -----------------------------

class Simulation:
    """Run bankroll simulations across multiple bets."""

    def __init__(self, initial_bankroll: float = 1000.0):
        self.bankroll = initial_bankroll
        self.history: List[Dict] = []

    def place_bet(self, prob_win: float, odds: float,
                  strategy: str = "kelly", max_fraction: float = 0.05,
                  outcome: bool = None):
        if strategy == "kelly":
            fraction = min(kelly_fraction(prob_win, odds), max_fraction)
        else:
            fraction = max_fraction

        stake = self.bankroll * fraction
        ev = expected_value(prob_win, odds, stake)
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

# -----------------------------
# Duplicate File Handling
# -----------------------------

duplicate_files: List[str] = []

def track_duplicate(file_path: str, timestamp: str) -> str:
    base, ext = os.path.splitext(file_path)
    ts_file = f"{base}_{timestamp}{ext}"
    if os.path.exists(ts_file):
        duplicate_files.append(ts_file)
    return ts_file

def cleanup_duplicates():
    for f in duplicate_files:
        try:
            os.remove(f)
            logger.info(f"🗑️ Deleted duplicate file: {f}")
        except Exception as e:
            logger.error(f"❌ Failed to delete {f}: {e}")
            
# ============================================================
def ensure_columns(df: pd.DataFrame, required_cols: list, label: str = "dataframe") -> pd.DataFrame:
    for col in required_cols:
        if col not in df.columns:
            df[col] = None
            logger.warning(f"⚠️ Added missing column '{col}' to {label}")
    return df