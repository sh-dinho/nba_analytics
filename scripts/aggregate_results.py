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

    # Prepare plot
    plt.figure(figsize=(10, 6))

    for m in model_types:
        path = f"results/picks_bankroll_{m}.csv"
        if not os.path.exists(path):
            print(f"⚠️ Skipping {m}, no results file found.")
            continue

        df = pd.read_csv(path)
        final_bankroll = df.iloc[-1]["bankroll"]
        total_bets = len(df)
        wins = sum(df["won"])
        win_rate = wins / total_bets if total_bets > 0 else 0
        avg_ev = df["EV"].mean()
        avg_stake = df["stake"].mean()

        summaries.append({
            "Model": m,
            "Final_Bankroll": final_bankroll,
            "Total_Bets": total_bets,
            "Win_Rate": win_rate,
            "Avg_EV": avg_ev,
            "Avg_Stake": avg_stake
        })

        # Plot bankroll trajectory
        plt.plot(df.index, df["bankroll"], label=m)

    if summaries:
        summary_df = pd.DataFrame(summaries)
        os.makedirs("results", exist_ok=True)
        summary_df.to_csv("results/combined_summary.csv", index=False)
        print("📊 Combined summary saved to results/combined_summary.csv")

        # Save chart
        plt.title("Bankroll Trajectories by Model")
        plt.xlabel("Bet Index")
        plt.ylabel("Bankroll")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("results/bankroll_comparison.png")
        print("📈 Bankroll comparison chart saved to results/bankroll_comparison.png")
    else:
        print("❌ No results found to aggregate.")

if __name__ == "__main__":
    main()