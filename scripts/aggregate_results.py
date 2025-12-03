# ============================================================
# File: scripts/aggregate_results.py
# Purpose: Merge results from all model types into one summary CSV + chart
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    model_types = ["logistic", "xgb", "nn"]
    summaries = []

    # Prepare plot for bankroll trajectories
    plt.figure(figsize=(10, 6))

    for m in model_types:
        path = f"results/picks_bankroll_{m}.csv"
        if not os.path.exists(path):
            print(f"⚠️ Skipping {m}, no results file found.")
            continue

        df = pd.read_csv(path)
        df = df.rename(columns=str.lower)  # normalize headers

        if df.empty:
            print(f"⚠️ Skipping {m}, file is empty.")
            continue

        # Safe extraction of metrics
        final_bankroll = df["bankroll"].iloc[-1] if "bankroll" in df else 0
        total_bets = len(df)
        wins = df["won"].sum() if "won" in df else 0
        win_rate = wins / total_bets if total_bets > 0 else 0
        avg_ev = df["ev"].mean() if "ev" in df else 0
        avg_stake = df["stake"].mean() if "stake" in df else 0

        summaries.append({
            "Model": m,
            "Final_Bankroll": round(final_bankroll, 2),
            "Total_Bets": total_bets,
            "Win_Rate": round(win_rate, 3),
            "Avg_EV": round(avg_ev, 3),
            "Avg_Stake": round(avg_stake, 2)
        })

        # Plot bankroll trajectory
        if "bankroll" in df:
            plt.plot(df.index, df["bankroll"], label=f"{m} bankroll")

        # Optional: overlay cumulative win rate trend
        if "won" in df:
            cum_wins = df["won"].cumsum()
            cum_win_rate = cum_wins / (df.index + 1)
            plt.plot(df.index, cum_win_rate * final_bankroll, linestyle="--", alpha=0.6,
                     label=f"{m} win-rate trend (scaled)")

    if summaries:
        summary_df = pd.DataFrame(summaries)
        os.makedirs("results", exist_ok=True)
        summary_df.to_csv("results/combined_summary.csv", index=False)
        print("📊 Combined summary saved to results/combined_summary.csv")

        # Save chart
        plt.title("Bankroll Trajectories by Model")
        plt.xlabel("Bet Number")
        plt.ylabel("Bankroll ($)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("results/bankroll_comparison.png")
        print("📈 Bankroll comparison chart saved to results/bankroll_comparison.png")
    else:
        print("❌ No results found to aggregate.")

if __name__ == "__main__":
    main()