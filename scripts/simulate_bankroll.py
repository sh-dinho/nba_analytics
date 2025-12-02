# File: scripts/simulate_bankroll.py

import pandas as pd
import numpy as np

def simulate_bankroll(
    df,
    strategy="kelly",
    max_fraction=0.05,
    initial=1000.0,
    n_sims=10,
    seed=None
):
    """
    Monte Carlo bankroll simulation.
    df must contain:
        - decimal_odds
        - win_prob (or prob / pred_home_win_prob)
        - ev (optional but useful)
    """

    # -----------------------------
    # Safety & column handling
    # -----------------------------
    if df is None or len(df) == 0:
        return [], {
            "final_bankroll_mean": initial,
            "roi": 0.0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0
        }

    df = df.copy()

    # Accept pipeline column names
    if "prob" not in df.columns:
        if "win_prob" in df.columns:
            df["prob"] = df["win_prob"]
        elif "pred_home_win_prob" in df.columns:
            df["prob"] = df["pred_home_win_prob"]
        else:
            raise KeyError("df must contain 'prob', 'win_prob', or 'pred_home_win_prob'")

    rng = np.random.default_rng(seed)  # reproducible RNG

    all_metrics = []
    all_trajectories = []

    # -----------------------------------------
    # Precompute b = odds - 1 for efficiency
    # -----------------------------------------
    df["b"] = df["decimal_odds"] - 1

    for sim in range(n_sims):
        bankroll = initial
        trajectory = []
        wins = 0
        losses = 0

        for _, row in df.iterrows():
            p = row["prob"]
            b = row["b"]

            # ------------- Kelly stake fraction -------------
            if strategy == "kelly":
                if b <= 0:
                    f_kelly = 0
                else:
                    # Kelly formula: f = (bp - q) / b
                    q = 1 - p
                    f_kelly = (b * p - q) / b

                # Clamp
                f = max(0, min(f_kelly, max_fraction))
            else:
                f = max_fraction

            stake = bankroll * f

            # ------------- Simulate outcome -------------
            won = rng.random() < p
            bankroll = bankroll + stake * b if won else bankroll - stake
            bankroll = max(bankroll, 0)

            if won:
                wins += 1
            else:
                losses += 1

            trajectory.append(bankroll)

        # -------- Metrics for this simulation --------
        metrics = {
            "final_bankroll": bankroll,
            "roi": (bankroll - initial) / initial,
            "wins": wins,
            "losses": losses,
            "win_rate": wins / (wins + losses) if (wins + losses) > 0 else 0
        }

        all_metrics.append(metrics)
        all_trajectories.append(trajectory)

    # -----------------------------
    # Aggregate metrics
    # -----------------------------
    metrics_df = pd.DataFrame(all_metrics)
    avg_metrics = metrics_df.mean(numeric_only=True).to_dict()
    avg_metrics["final_bankroll_mean"] = avg_metrics.pop("final_bankroll", None)

    return all_trajectories, avg_metrics