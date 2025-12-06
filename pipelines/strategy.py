# ============================================================
# File: pipelines/strategy.py
# Purpose: Bankroll management (Kelly / Flat) + simulations
# ============================================================

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict

from core.log_config import setup_logger
from core.exceptions import PipelineError, DataError
from core.config import (
    DEFAULT_BANKROLL,
    MAX_KELLY_FRACTION,
    BASE_RESULTS_DIR,
    PIPELINE_SUMMARY_FILE
)

# Optional Telegram utilities (safe import)
try:
    from core.telegram_utils import send_message, send_photo
except Exception:
    def send_message(*args, **kwargs):
        pass
    def send_photo(*args, **kwargs):
        pass

logger = setup_logger("strategy")

# ============================================================
# Utility Functions
# ============================================================

def Expected_Value(prob_win: float, odds: float, stake: float = 1.0) -> float:
    if not (0 <= prob_win <= 1):
        raise DataError("Probability must be between 0 and 1")
    if odds <= 0:
        raise DataError("Odds must be greater than 0")

    prob_loss = 1 - prob_win
    profit = (odds * stake) - stake
    return (prob_win * profit) - (prob_loss * stake)


def Kelly_Fraction(prob_win: float, odds: float) -> float:
    if not (0 <= prob_win <= 1):
        raise DataError("Probability must be between 0 and 1")

    b = odds - 1
    if b <= 0:
        return 0

    p = prob_win
    q = 1 - p
    k = (b * p - q) / b
    return max(0, k)


def Update_Bankroll(bankroll: float, stake: float, won: bool, odds: float) -> float:
    """Compute updated bankroll after the result."""
    if won:
        return bankroll + ((odds * stake) - stake)
    return bankroll - stake


def Implied_Probability(odds: float) -> float:
    return 1/odds if odds > 0 else 0


# ============================================================
# Simulation Class
# ============================================================

class Simulation:
    """Simulate series of bets under a given strategy."""

    def __init__(self, initial_bankroll: float = DEFAULT_BANKROLL):
        self.bankroll = initial_bankroll
        self.trajectory = [initial_bankroll]
        self.history: List[Dict] = []

    def place_bet(self, prob_win: float, odds: float,
                  strategy: str = "kelly",
                  max_fraction: float = MAX_KELLY_FRACTION,
                  outcome: bool = None):
        """Place one bet under Kelly or Flat strategy."""

        # Determine stake fraction
        if strategy == "kelly":
            fraction = min(Kelly_Fraction(prob_win, odds), max_fraction)
        elif strategy == "flat":
            # Flat = fixed fraction, not based on Kelly
            fraction = max_fraction
        else:
            raise PipelineError(f"Unknown strategy: {strategy}")

        stake = self.bankroll * fraction
        ev = Expected_Value(prob_win, odds, stake)

        # Determine outcome (stochastic if None)
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

    def run(self, bets: List[Dict], strategy: str, max_fraction: float):
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
        """Final bankroll + aggregate statistics."""
        total = len(self.history)
        wins = sum(1 for h in self.history if h["won"])
        return {
            "Final_Bankroll": self.bankroll,
            "Total_Bets": total,
            "Win_Rate": wins / total if total else 0,
            "Avg_EV": np.mean([h["EV"] for h in self.history]) if total else 0,
            "Avg_Stake": np.mean([h["stake"] for h in self.history]) if total else 0
        }


# ============================================================
# Apply Strategy to Model Predictions
# ============================================================

def apply_strategy(predictions_df: pd.DataFrame,
                   threshold: float = 0.6,
                   strategy: str = "kelly",
                   max_fraction: float = MAX_KELLY_FRACTION,
                   initial_bankroll: float = DEFAULT_BANKROLL,
                   use_fair_odds: bool = True):

    if "pred_home_win_prob" not in predictions_df.columns:
        raise DataError("Missing column: 'pred_home_win_prob'")

    BASE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    sim = Simulation(initial_bankroll)

    bets = []
    for _, row in predictions_df.iterrows():
        p = row["pred_home_win_prob"]
        if p < threshold:
            continue

        if use_fair_odds:
            odds = 1 / p
        else:
            odds = 2.0

        bets.append({"prob_win": p, "odds": odds})

    sim.run(bets, strategy=strategy, max_fraction=max_fraction)
    metrics = sim.summary()

    picks_df = pd.DataFrame(sim.history)
    picks_df["bankroll_after"] = picks_df["bankroll"]

    # Save trajectory
    plt.plot(sim.trajectory)
    plt.title(f"Bankroll Trajectory ({strategy})")
    plt.xlabel("Bet #")
    plt.ylabel("Bankroll")
    plot_path = BASE_RESULTS_DIR / f"trajectory_{strategy}.png"
    plt.savefig(plot_path)
    plt.close()

    logger.info(
        f"Strategy={strategy} | Final=${metrics['Final_Bankroll']:.2f} | ROI={(metrics['Final_Bankroll']/initial_bankroll - 1):.2%}"
    )

    return picks_df, metrics, sim.trajectory


# ============================================================
# Strategy Comparison
# ============================================================

def compare_strategies(predictions_df: pd.DataFrame, strategies=["kelly", "flat"]):
    results = {}
    for s in strategies:
        _, metrics, _ = apply_strategy(predictions_df, strategy=s)
        results[s] = metrics
    return pd.DataFrame(results).T


# ============================================================
# Monte Carlo Simulation
# ============================================================

def monte_carlo_strategy(predictions_df: pd.DataFrame,
                         n_runs: int = 1000,
                         strategy: str = "kelly",
                         threshold: float = 0.6):

    results = []

    for _ in range(n_runs):
        _, metrics, _ = apply_strategy(
            predictions_df,
            threshold=threshold,
            strategy=strategy
        )
        results.append(metrics["Final_Bankroll"])

    mean = np.mean(results)
    std = np.std(results)

    # Save histogram
    plt.hist(results, bins=30)
    plt.title(f"Monte Carlo ({strategy})")
    plt.xlabel("Final Bankroll")
    plt.ylabel("Frequency")
    plot_path = BASE_RESULTS_DIR / f"mc_{strategy}.png"
    plt.savefig(plot_path)
    plt.close()

    return {
        "mean_final_bankroll": mean,
        "std_dev": std
    }


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bankroll strategy evaluation")

    parser.add_argument("--predictions", required=True, help="CSV with predictions")
    parser.add_argument("--strategy", default="kelly", choices=["kelly", "flat"])
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--notify", action="store_true")
    parser.add_argument("--monte-carlo", type=int, help="Run MC for one strategy")
    parser.add_argument("--multi-monte", type=int, help="Run MC for Kelly + Flat")

    args = parser.parse_args()

    df = pd.read_csv(args.predictions)

    # ---------------------------------------------------------
    # Multi Monte Carlo
    # ---------------------------------------------------------
    if args.multi_monte:

        rows = []
        for strat in ["kelly", "flat"]:
            stats = monte_carlo_strategy(df, args.multi_monte, strategy=strat)
            logger.info(f"{strat.upper()} MC → Mean={stats['mean_final_bankroll']:.2f}, Std={stats['std_dev']:.2f}")

            rows.append({
                "strategy": strat,
                "mean_final_bankroll": stats["mean_final_bankroll"],
                "std_dev": stats["std_dev"]
            })

            if args.notify:
                send_message(f"Monte Carlo ({strat})\nMean={stats['mean_final_bankroll']:.2f}\nStd={stats['std_dev']:.2f}")
                send_photo(str(BASE_RESULTS_DIR / f"mc_{strat}.png"))

        out = pd.DataFrame(rows)
        out.to_csv(PIPELINE_SUMMARY_FILE, mode="a", header=not PIPELINE_SUMMARY_FILE.exists(), index=False)
        logger.info(f"Saved MC summary → {PIPELINE_SUMMARY_FILE}")

    # ---------------------------------------------------------
    # Single Monte Carlo
    # ---------------------------------------------------------
    elif args.monte_carlo:
        stats = monte_carlo_strategy(df, args.monte_carlo, strategy=args.strategy)
        logger.info(f"{args.strategy.upper()} MC → Mean={stats['mean_final_bankroll']:.2f}, Std={stats['std_dev']:.2f}")

        if args.notify:
            send_message(f"Monte Carlo ({args.strategy})\nMean={stats['mean_final_bankroll']:.2f}\nStd={stats['std_dev']:.2f}")
            send_photo(str(BASE_RESULTS_DIR / f"mc_{args.strategy}.png"))

    # ---------------------------------------------------------
    # Regular strategy run
    # ---------------------------------------------------------
    else:
        picks, metrics, traj = apply_strategy(
            df,
            threshold=args.threshold,
            strategy=args.strategy
        )

        logger.info("Completed strategy run.")
        logger.info(metrics)

        if args.notify:
            send_message(f"Strategy={args.strategy}\nFinal Bankroll=${metrics['Final_Bankroll']:.2f}")
            send_photo(str(BASE_RESULTS_DIR / f"trajectory_{args.strategy}.png"))
