# ============================================================
# File: pipelines/strategy.py
# Purpose: Apply bankroll management strategies (Kelly or Flat) with simulation support
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict
from core.log_config import setup_logger
from core.exceptions import PipelineError, DataError
from core.config import DEFAULT_BANKROLL, MAX_KELLY_FRACTION, BASE_RESULTS_DIR

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


# --- Pipeline Integration Functions ---

def apply_strategy(predictions_df: pd.DataFrame,
                   threshold: float = 0.6,
                   strategy: str = "kelly",
                   max_fraction: float = MAX_KELLY_FRACTION,
                   initial_bankroll: float = DEFAULT_BANKROLL,
                   use_fair_odds: bool = True):
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
        odds = (1 / prob) if use_fair_odds else 2.0
        bets.append({"prob_win": prob, "odds": odds})

    sim.run(bets, strategy=strategy, max_fraction=max_fraction)
    metrics = sim.summary()

    picks_df = pd.DataFrame(sim.history)
    picks_df["bankroll_after"] = [h["bankroll"] for h in sim.history]

    # Save bankroll trajectory plot
    plt.plot(sim.trajectory)
    plt.title(f"Bankroll Trajectory ({strategy})")
    plt.xlabel("Bet Number")
    plt.ylabel("Bankroll")
    plot_path = BASE_RESULTS_DIR / f"bankroll_{strategy}.png"
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"📊 Saved bankroll trajectory plot → {plot_path}")

    logger.info(f"Strategy applied: {strategy}, Final bankroll={metrics['Final_Bankroll']:.2f}, ROI={(metrics['Final_Bankroll']-initial_bankroll)/initial_bankroll:.2%}")
    return picks_df, metrics, sim.trajectory


def compare_strategies(predictions_df: pd.DataFrame, strategies=["kelly", "flat"]):
    """Compare multiple strategies side-by-side."""
    results = {}
    for strat in strategies:
        _, metrics, _ = apply_strategy(predictions_df, strategy=strat)
        results[strat] = metrics
    return pd.DataFrame(results).T


def monte_carlo_strategy(predictions_df: pd.DataFrame,
                         n_runs: int = 1000,
                         strategy: str = "kelly",
                         threshold: float = 0.6):
    """Run Monte Carlo simulations of bankroll outcomes."""
    final_bankrolls = []
    for _ in range(n_runs):
        _, metrics, _ = apply_strategy(predictions_df, strategy=strategy, threshold=threshold)
        final_bankrolls.append(metrics["Final_Bankroll"])

    mean_bankroll = np.mean(final_bankrolls)
    std_bankroll = np.std(final_bankrolls)

    # Save histogram
    plt.hist(final_bankrolls, bins=30, alpha=0.7)
    plt.title(f"Monte Carlo Simulation ({strategy})")
    plt.xlabel("Final Bankroll")
    plt.ylabel("Frequency")
    plot_path = BASE_RESULTS_DIR / f"monte_carlo_{strategy}.png"
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"📊 Saved Monte Carlo histogram → {plot_path}")

    return {"mean_final_bankroll": mean_bankroll, "std_dev": std_bankroll}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply bankroll management strategy to predictions")
    parser.add_argument("--predictions", type=str, required=True, help="Path to predictions CSV")
    parser.add_argument("--strategy", type=str, choices=["kelly", "flat"], default="kelly", help="Strategy to apply")
    parser.add_argument("--threshold", type=float, default=0.6, help="Probability threshold for betting")
    parser.add_argument("--season", type=str, default="aggregate", help="Season tag for entries")
    parser.add_argument("--notes", type=str, default="strategy run", help="Optional notes to annotate entries")
    parser.add_argument("--notify", action="store_true", help="Send results to Telegram")
    parser.add_argument("--monte-carlo", type=int, help="Run Monte Carlo simulation with N runs")
    parser.add_argument("--multi-monte-carlo", type=int, help="Run Monte Carlo for both Kelly and Flat strategies")
    args = parser.parse_args()

    df = pd.read_csv(args.predictions)

    if args.multi_monte_carlo:
        strategies = ["kelly", "flat"]
        summary_rows = []
        for strat in strategies:
            stats = monte_carlo_strategy(df, n_runs=args.multi_monte_carlo,
                                         strategy=strat, threshold=args.threshold)
            logger.info(f"Monte Carlo ({strat}) — Mean bankroll={stats['mean_final_bankroll']:.2f}, Std Dev={stats['std_dev']:.2f}")

            run_time = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            summary_rows.append({
                "timestamp": run_time,
                "season": args.season,
                "target": "multi_monte_carlo_strategy",
                "model_type": strat,
                "notes": args.notes,
                "mean_final_bankroll": stats["mean_final_bankroll"],
                "std_dev": stats["std_dev"]
            })

            if args.notify:
                send_message(f"🤖 Monte Carlo ({strat}) — Mean bankroll={stats['mean_final_bankroll']:.2f}, Std Dev={stats['std_dev']:.2f}")
                plot_path = BASE_RESULTS_DIR / f"monte_carlo_{strat}.png"
                send_photo(str(plot_path), caption=f"📊 Monte Carlo Simulation ({strat})")

        # Append both strategies to pipeline_summary.csv
        summary_df = pd.DataFrame(summary_rows)
        if not PIPELINE_SUMMARY_FILE.exists():
            summary_df.to_csv(PIPELINE_SUMMARY_FILE, index=False)
        else:
            summary_df.to_csv(PIPELINE_SUMMARY_FILE, mode="a", header=False, index=False)
        logger.info(f"📑 Multi-strategy Monte Carlo results appended to {PIPELINE_SUMMARY_FILE}")

    elif args.monte_carlo:
        # Single-strategy Monte Carlo (existing code)
        stats = monte_carlo_strategy(df, n_runs=args.monte_carlo, strategy=args.strategy, threshold=args.threshold)
        ...
    else:
        # Default single strategy run
        apply_strategy(df,
                       threshold=args.threshold,
                       strategy=args.strategy,
                       season=args.season,
                       notes=args.notes,
                       notify=args.notify)
