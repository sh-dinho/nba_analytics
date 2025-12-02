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
import os
from typing import List, Tuple, Dict, Any, Optional

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


# ================================================================
# Core bankroll simulator
# ================================================================

def simulate_bankroll(
    df: pd.DataFrame,
    strategy: str = "kelly",
    initial: float = 1000.0,
    max_fraction: float = 0.05,
    runs: int = 1,
    seed: Optional[int] = None,
    show_progress: bool = True,
    use_actual_outcomes: bool = True
) -> Tuple[List[List[float]], Dict[str, Any]]:
    """
    Simulate bankroll trajectories.

    Parameters
    ----------
    df : DataFrame
        Required columns:
        - "decimal_odds": float
        - "prob": model probability of home team winning

        Optional:
        - "home_win": if present, used for real-outcome mode

    strategy : str
        "kelly" or "flat"
    initial : float
        Starting bankroll.
    max_fraction : float
        Maximum fraction used either as a Kelly cap or flat-bet fraction.
    runs : int
        Number of Monte Carlo runs.
    seed : int or None
        Reproducibility seed.
    show_progress : bool
        Whether to show tqdm bar.
    use_actual_outcomes : bool
        If True and "home_win" column is available → uses real outcomes instead of stochastic sampling.

    Returns
    -------
    all_trajectories : list[list[float]]
    metrics : dict
    """

    required = {"decimal_odds", "prob"}
    if not required.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required}")

    if seed is not None:
        np.random.seed(seed)

    actual_available = use_actual_outcomes and ("home_win" in df.columns)

    # Extract vectors for speed
    probs = df["prob"].astype(float).values
    odds = df["decimal_odds"].astype(float).values
    b_vals = odds - 1.0

    if actual_available:
        actual = df["home_win"].astype(int).values

    # ================================================================
    # Simulation loop
    # ================================================================
    all_trajectories = []
    final_bankrolls = []
    win_counts = []
    loss_counts = []

    iterator = range(runs)
    if show_progress and runs > 1:
        iterator = tqdm(iterator, desc="Simulating runs", unit="run")

    for _ in iterator:
        bankroll = initial
        trajectory = []
        wins = losses = 0

        for i in range(len(df)):
            p = probs[i]
            o = odds[i]
            b = b_vals[i]

            # ----------------------------
            # Stake sizing
            # ----------------------------
            if strategy == "kelly":
                if b <= 0:
                    fraction = 0.0
                else:
                    k = ((b * p) - (1 - p)) / b
                    fraction = max(0.0, min(k, max_fraction))
            else:
                # Flat betting
                fraction = max_fraction

            stake = bankroll * fraction

            # ----------------------------
            # Determine outcome
            # ----------------------------
            if actual_available:
                won = bool(actual[i])
            else:
                won = np.random.rand() < p

            if won:
                bankroll += stake * b
                wins += 1
            else:
                bankroll -= stake
                losses += 1

            bankroll = max(bankroll, 0)  # avoid tiny negative float drift
            trajectory.append(bankroll)

        all_trajectories.append(trajectory)
        final_bankrolls.append(bankroll)
        win_counts.append(wins)
        loss_counts.append(losses)

    # ================================================================
    # Aggregate metrics
    # ================================================================
    avg_final = float(np.mean(final_bankrolls))
    roi = (avg_final - initial) / initial if initial else 0.0
    avg_wins = float(np.mean(win_counts))
    avg_losses = float(np.mean(loss_counts))
    win_rate = avg_wins / (avg_wins + avg_losses) if (avg_wins + avg_losses) else 0.0

    metrics = {
        "final_bankroll_mean": avg_final,
        "final_bankroll_std": float(np.std(final_bankrolls)),
        "roi": roi,
        "wins_mean": avg_wins,
        "losses_mean": avg_losses,
        "win_rate": win_rate,
        "runs": runs,
        "initial": initial,
        "strategy": strategy,
        "max_fraction": max_fraction,
        "seed": seed,
        "use_actual_outcomes": actual_available,
    }

    return all_trajectories, metrics


# ================================================================
# Plotting
# ================================================================
def plot_trajectories(all_trajectories: List[List[float]]):
    plt.figure(figsize=(8, 4))
    for traj in all_trajectories:
        plt.plot(traj, alpha=0.5)
    plt.title("Bankroll Trajectories")
    plt.xlabel("Bet #")
    plt.ylabel("Bankroll ($)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ================================================================
# CLI for standalone usage
# ================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate bankroll trajectory")
    parser.add_argument("--csv", type=str, default="results/picks.csv", help="Path to picks CSV file")
    parser.add_argument("--strategy", choices=["kelly", "flat"], default="kelly")
    parser.add_argument("--initial", type=float, default=1000)
    parser.add_argument("--max_fraction", type=float, default=0.05)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--ignore-actual", action="store_true",
                        help="Force Monte Carlo even if actual outcomes are available")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    trajectories, metrics = simulate_bankroll(
        df,
        strategy=args.strategy,
        initial=args.initial,
        max_fraction=args.max_fraction,
        runs=args.runs,
        seed=args.seed,
        show_progress=not args.no_progress,
        use_actual_outcomes=not args.ignore_actual
    )

    logger.info("\n=== SIMULATION METRICS ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            logger.info(f"{k}: {v:.4f}")
        else:
            logger.info(f"{k}: {v}")

    # Save results
    os.makedirs("results", exist_ok=True)
    ts_metrics_file = f"results/simulation_metrics_{_timestamp()}.json"
    with open(ts_metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"🧾 Metrics saved to {ts_metrics_file}")

    if args.plot:
        plot_trajectories(trajectories)
