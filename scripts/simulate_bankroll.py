# File: scripts/simulate_bankroll.py
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import sys
import logging
import json
from datetime import datetime
from tqdm import tqdm

# ----------------------------
# Logging setup
# ----------------------------
logger = logging.getLogger("simulate_bankroll")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
logger.addHandler(handler)


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def simulate_bankroll(
    df: pd.DataFrame,
    strategy: str = "kelly",
    initial: float = 1000,
    max_fraction: float = 0.05,
    runs: int = 1,
    seed: int = None,
    show_progress: bool = True
):
    """
    Simulate bankroll trajectory given predictions and odds.

    Parameters:
        df: DataFrame with columns ["decimal_odds", "prob", "ev"]
        strategy: "kelly" or "flat"
        initial: starting bankroll
        max_fraction: cap on Kelly fraction or flat fraction
        runs: number of Monte Carlo runs
        seed: random seed for reproducibility
        show_progress: whether to show tqdm progress bar

    Returns:
        all_trajectories (list of lists), metrics_dict (aggregated metrics)
    """
    if seed is not None:
        np.random.seed(seed)

    all_trajectories = []
    final_bankrolls = []
    win_counts = []
    loss_counts = []

    iterator = range(runs)
    if show_progress and runs > 1:
        iterator = tqdm(iterator, desc="Simulating runs", unit="run")

    for run in iterator:
        bankroll = initial
        trajectory = []
        wins = losses = 0

        for _, row in df.iterrows():
            p = float(row["prob"])
            o = float(row["decimal_odds"])
            b = o - 1.0

            # Stake sizing
            if strategy == "kelly":
                kelly = ((b * p) - (1 - p)) / b if b > 0 else 0
                fraction = max(0.0, min(kelly, max_fraction))
            elif strategy == "flat":
                fraction = max_fraction
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            stake = bankroll * fraction

            # Random outcome based on probability
            won = np.random.rand() < p
            if won:
                bankroll += stake * b
                wins += 1
            else:
                bankroll -= stake
                losses += 1

            trajectory.append(bankroll)

        all_trajectories.append(trajectory)
        final_bankrolls.append(bankroll)
        win_counts.append(wins)
        loss_counts.append(losses)

    # Aggregate metrics
    avg_final = np.mean(final_bankrolls)
    roi = (avg_final - initial) / initial if initial else 0.0
    avg_wins = np.mean(win_counts)
    avg_losses = np.mean(loss_counts)
    win_rate = avg_wins / (avg_wins + avg_losses) if (avg_wins + avg_losses) else 0.0

    metrics = {
        "final_bankroll_mean": avg_final,
        "final_bankroll_std": np.std(final_bankrolls),
        "roi_mean": roi,
        "wins_mean": avg_wins,
        "losses_mean": avg_losses,
        "win_rate_mean": win_rate,
        "runs": runs,
        "initial": initial,
        "strategy": strategy,
        "max_fraction": max_fraction,
        "seed": seed
    }

    return all_trajectories, metrics


def plot_trajectories(all_trajectories):
    """Plot bankroll trajectories across runs."""
    plt.figure(figsize=(8, 4))
    for traj in all_trajectories:
        plt.plot(traj, alpha=0.5)
    plt.title("Bankroll Trajectories")
    plt.xlabel("Bet #")
    plt.ylabel("Bankroll ($)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate bankroll trajectory")
    parser.add_argument("--csv", type=str, default="results/picks.csv", help="Path to picks CSV file")
    parser.add_argument("--strategy", choices=["kelly", "flat"], default="kelly", help="Betting strategy")
    parser.add_argument("--initial", type=float, default=1000, help="Initial bankroll")
    parser.add_argument("--max_fraction", type=float, default=0.05, help="Max fraction per bet")
    parser.add_argument("--runs", type=int, default=1, help="Number of Monte Carlo runs")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--plot", action="store_true", help="Plot bankroll trajectories")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bar")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    trajectories, metrics = simulate_bankroll(
        df,
        strategy=args.strategy,
        initial=args.initial,
        max_fraction=args.max_fraction,
        runs=args.runs,
        seed=args.seed,
        show_progress=not args.no_progress
    )

    logger.info("\n=== SIMULATION METRICS ===")
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            logger.info(f"{k}: {v:.4f}")
        else:
            logger.info(f"{k}: {v}")

    # Save timestamped backups
    os.makedirs("results", exist_ok=True)
    ts_metrics_file = f"results/simulation_metrics_{_timestamp()}.json"
    with open(ts_metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"🧾 Metrics saved to {ts_metrics_file}")

    if args.plot:
        plot_trajectories(trajectories)