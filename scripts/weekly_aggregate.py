# ============================================================
# File: scripts/weekly_aggregate.py
# Purpose: Aggregate 7-day results, track wins by model, add AI insight + chart
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import datetime

def main():
    model_types = ["logistic", "xgb", "nn"]
    summaries = []

    cutoff = datetime.datetime.now() - datetime.timedelta(days=7)

    plt.figure(figsize=(10, 6))

    for m in model_types:
        files = [f for f in os.listdir("results") if f.startswith(f"picks_bankroll_{m}")]
        files.sort()

        total_wins, total_bets, final_bankrolls = 0, 0, []
        daily_bankrolls = []

        for f in files:
            path = os.path.join("results", f)
            df = pd.read_csv(path)
            total_bets += len(df)
            total_wins += sum(df["won"])
            final_bankrolls.append(df.iloc[-1]["bankroll"])
            daily_bankrolls.append(df.iloc[-1]["bankroll"])

        if total_bets > 0:
            summaries.append({
                "Model": m,
                "Total_Bets": total_bets,
                "Total_Wins": total_wins,
                "Win_Rate": total_wins / total_bets,
                "Avg_Final_Bankroll": sum(final_bankrolls) / len(final_bankrolls)
            })
            # Plot weekly bankroll trend
            plt.plot(range(len(daily_bankrolls)), daily_bankrolls, marker="o", label=m)

    if summaries:
        summary_df = pd.DataFrame(summaries)
        os.makedirs("results", exist_ok=True)
        summary_df.to_csv("results/weekly_summary.csv", index=False)
        print("📊 Weekly summary saved to results/weekly_summary.csv")

        # AI insight
        best_model = summary_df.loc[summary_df["Avg_Final_Bankroll"].idxmax()]
        if best_model["Win_Rate"] > 0.55:
            ai_insight = f"🤖 Weekly AI Insight: {best_model['Model']} shows sustained edge with {best_model['Win_Rate']:.2%} win rate."
        else:
            ai_insight = f"🤖 Weekly AI Insight: No clear edge — performance varied across models."

        with open("results/weekly_ai_insight.txt", "w") as f:
            f.write(ai_insight)
        print(ai_insight)

        # Save chart
        plt.title("Weekly Bankroll Trends by Model")
        plt.xlabel("Day Index")
        plt.ylabel("Final Bankroll")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("results/weekly_bankroll_trends.png")
        print("📈 Weekly bankroll chart saved to results/weekly_bankroll_trends.png")
    else:
        print("❌ No weekly results found.")

if __name__ == "__main__":
    main()