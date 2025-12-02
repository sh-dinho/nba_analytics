# ============================================================
# File: scripts/monthly_aggregate.py
# Purpose: Aggregate monthly results, track wins by model, add AI insight + chart
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import re
from core.config import BASE_RESULTS_DIR, MONTHLY_SUMMARY_FILE 
# Note: BANKROLL_FILE_TEMPLATE is used to define the file *pattern*

def main():
    model_types = ["logistic", "xgb", "nn"]
    summaries = []

    # Current month cutoff based on run time
    now = datetime.datetime.now()
    current_month_str = now.strftime("%Y-%m") # e.g., 2025-12
    
    results_dir = BASE_RESULTS_DIR
    
    # Prepare plot
    plt.figure(figsize=(10, 6))

    for m in model_types:
        # Find all bankroll files that match the prefix
        files_in_dir = os.listdir(results_dir)
        bankroll_files = [
            f for f in files_in_dir 
            if f.startswith(f"picks_bankroll_{m}") and f.endswith(".csv")
        ]
        bankroll_files.sort()

        total_wins, total_bets = 0, 0
        final_bankrolls_daily = [] # Last bankroll value of each day/file processed
        
        print(f"Processing {m} model with {len(bankroll_files)} files...")

        for f in bankroll_files:
            path = os.path.join(results_dir, f)
            try:
                df = pd.read_csv(path)
            except pd.errors.EmptyDataError:
                print(f"  > Skipping empty file: {f}")
                continue
            
            # ------------------------------------------------
            # CRITICAL FIX: Filter by 'Date' column content
            # ------------------------------------------------
            df_month = pd.DataFrame() # Initialize empty

            if 'Date' in df.columns and not df['Date'].empty:
                # Assuming 'Date' column is in YYYY-MM-DD format (set in prediction_pipeline.py)
                df['Date'] = pd.to_datetime(df['Date'])
                df_month = df[df['Date'].dt.strftime('%Y-%m') == current_month_str]
            else:
                # Fallback: Check if the filename itself contains the month string
                # This logic is less reliable but covers cases where the internal date is missing
                if current_month_str in f:
                    df_month = df
                
            if not df_month.empty:
                total_bets += len(df_month)
                # 'won' is 1 for win, 0 for loss, -1 for pending (prediction mode)
                total_wins += df_month["won"].loc[df_month["won"] != -1].sum()
                
                # Use the last bankroll value of the file for the summary/plot point
                final_bankroll_value = df_month.iloc[-1]["bankroll"]
                final_bankrolls_daily.append(final_bankroll_value)

        if total_bets > 0:
            avg_final_bankroll = sum(final_bankrolls_daily) / len(final_bankrolls_daily)
            summaries.append({
                "Model": m,
                "Total_Bets": total_bets,
                "Total_Wins": total_wins,
                "Win_Rate": total_wins / total_bets,
                "Avg_Final_Bankroll": avg_final_bankroll
            })
            
            # Plot against day index (1, 2, 3...)
            plt.plot(range(1, len(final_bankrolls_daily) + 1), final_bankrolls_daily, marker="o", label=m)

    if summaries:
        summary_df = pd.DataFrame(summaries)
        
        # Save summary
        os.makedirs(os.path.dirname(MONTHLY_SUMMARY_FILE), exist_ok=True)
        summary_df.to_csv(MONTHLY_SUMMARY_FILE, index=False)
        
        # AI insight
        best_model = summary_df.loc[summary_df["Avg_Final_Bankroll"].idxmax()]
        if best_model["Win_Rate"] > 0.55:
            ai_insight = f"🤖 Monthly AI Insight: {best_model['Model']} shows sustained edge with {best_model['Win_Rate']:.2%} win rate."
        else:
            ai_insight = f"🤖 Monthly AI Insight: No clear edge — performance varied across models."

        # Save AI insight
        monthly_insight_file = os.path.join(os.path.dirname(MONTHLY_SUMMARY_FILE), "monthly_ai_insight.txt")
        with open(monthly_insight_file, "w") as f:
            f.write(ai_insight)

        # Save chart
        chart_file = os.path.join(os.path.dirname(MONTHLY_SUMMARY_FILE), "monthly_bankroll_comparison.png")
        plt.title(f"Bankroll Trends for {current_month_str} by Model")
        plt.xlabel("Day Index")
        plt.ylabel("Final Bankroll")
        plt.legend()
        plt.grid(True)
        plt.savefig(chart_file)
        plt.close()
        
        print(f"📊 Monthly summary saved to {MONTHLY_SUMMARY_FILE}")
        print(f"📈 Chart saved to {chart_file}")
        print(ai_insight)
    else:
        print(f"⚠️ No bankroll data found for month {current_month_str} to aggregate.")

if __name__ == "__main__":
    main()