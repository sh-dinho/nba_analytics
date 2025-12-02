# ============================================================
# File: scripts/aggregate_results.py
# Purpose: Merge results from all model types into one summary CSV
# ============================================================

import os
import pandas as pd

def main():
    model_types = ["logistic", "xgb", "nn"]
    summaries = []

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

    if summaries:
        summary_df = pd.DataFrame(summaries)
        os.makedirs("results", exist_ok=True)
        summary_df.to_csv("results/combined_summary.csv", index=False)
        print("📊 Combined summary saved to results/combined_summary.csv")
    else:
        print("❌ No results found to aggregate.")

if __name__ == "__main__":
    main()