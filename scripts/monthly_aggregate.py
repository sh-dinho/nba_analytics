# ============================================================
# File: scripts/monthly_aggregate.py
# Purpose: Aggregate monthly results, track wins by model, add AI insight + chart
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import datetime

def main():
    model_types = ["logistic", "xgb", "nn"]
    summaries = []

    # Current month cutoff
    now = datetime.datetime.now()
    month_str = now.strftime("%Y%m")  # e.g., 202512

    plt.figure(figsize=(10, 6))

    for m in model_types:
        files = [f for f in os.listdir("results") if f.startswith(f"picks_bankroll_{m}")]
        files.sort()

        total_wins, total_bets, final_bankrolls = 0, 0, []
        daily_bankrolls = []

        for f in files:
            # Expect filenames like picks_bankroll_model_YYYYMMDD.csv
            if month_str in f:
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
            plt.plot(range(len(daily_bankrolls)), daily_bankrolls, marker="o", label=m)

    if summaries:
        summary_df = pd.DataFrame(summaries)
        os.makedirs("results", exist_ok=True)
        summary_df.to_csv("results/monthly_summary.csv", index=False)
        print("📊 Monthly summary saved to results/monthly_summary.csv")

        # AI insight
        best_model = summary_df.loc[summary_df["Avg_Final_Bankroll"].idxmax()]
        if best_model["Win_Rate"] > 0.55:
            ai_insight = f"🤖 Monthly AI Insight: {best_model['Model']} shows sustained edge with {best_model['Win_Rate']:.2%} win rate."
        else:
            ai_insight = f"🤖 Monthly AI Insight: No clear edge — performance varied across models."

        with open("results/monthly_ai_insight.txt", "w") as f:
            f.write(ai_insight)
        print(ai_insight)

        # Save chart
        plt.title("Monthly Bankroll Trends by Model")
        plt.xlabel("Day Index")
        plt.ylabel("Final Bankroll")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("results/monthly_bankroll_trends.png")
        print("📈 Monthly bankroll chart saved to results/monthly_bankroll_trends.png")
    else:
        print("❌ No monthly results found.")

if __name__ == "__main__":
    main()