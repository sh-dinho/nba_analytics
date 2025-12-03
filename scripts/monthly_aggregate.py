# ============================================================
# File: scripts/monthly_aggregate.py
# Purpose: Aggregate monthly results, track wins by model, add AI insight + chart
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from core.config import BASE_RESULTS_DIR, MONTHLY_SUMMARY_FILE
from core.log_config import setup_logger
from core.exceptions import PipelineError, DataError

logger = setup_logger("monthly_aggregate")


def main():
    model_types = ["logistic", "xgb", "nn"]
    summaries = []

    # Current month cutoff based on run time
    now = datetime.datetime.now()
    current_month_str = now.strftime("%Y-%m")  # e.g., 2025-12

    results_dir = BASE_RESULTS_DIR

    # Prepare plot
    plt.figure(figsize=(10, 6))

    for m in model_types:
        # Find all bankroll files that match the prefix
        try:
            files_in_dir = os.listdir(results_dir)
        except Exception as e:
            raise PipelineError(f"Failed to list results directory {results_dir}: {e}")

        bankroll_files = [
            f for f in files_in_dir
            if f.startswith(f"picks_bankroll_{m}") and f.endswith(".csv")
        ]
        bankroll_files.sort()

        total_wins, total_bets = 0, 0
        final_bankrolls_daily = []  # Last bankroll value of each day/file processed

        logger.info(f"Processing {m} model with {len(bankroll_files)} files...")

        for f in bankroll_files:
            path = os.path.join(results_dir, f)
            try:
                df = pd.read_csv(path)
            except pd.errors.EmptyDataError:
                logger.warning(f"Skipping empty file: {f}")
                continue
            except Exception as e:
                logger.warning(f"Failed to read {f}: {e}")
                continue

            df_month = pd.DataFrame()

            if "Date" in df.columns and not df["Date"].empty:
                try:
                    df["Date"] = pd.to_datetime(df["Date"])
                    df_month = df[df["Date"].dt.strftime("%Y-%m") == current_month_str]
                except Exception as e:
                    logger.warning(f"Failed to parse Date column in {f}: {e}")
            else:
                # Fallback: Check if the filename itself contains the month string
                if current_month_str in f:
                    df_month = df

            if not df_month.empty:
                total_bets += len(df_month)
                if "won" not in df_month.columns or "bankroll" not in df_month.columns:
                    logger.warning(f"Missing required columns in {f}. Skipping.")
                    continue

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
            plt.plot(range(1, len(final_bankrolls_daily) + 1),
                     final_bankrolls_daily, marker="o", label=m)

    if summaries:
        summary_df = pd.DataFrame(summaries)

        # Save summary
        os.makedirs(os.path.dirname(MONTHLY_SUMMARY_FILE), exist_ok=True)
        summary_df.to_csv(MONTHLY_SUMMARY_FILE, index=False)
        logger.info(f"📊 Monthly summary saved to {MONTHLY_SUMMARY_FILE}")

        # AI insight
        try:
            best_model = summary_df.loc[summary_df["Avg_Final_Bankroll"].idxmax()]
            if best_model["Win_Rate"] > 0.55:
                ai_insight = (
                    f"🤖 Monthly AI Insight: {best_model['Model']} shows sustained edge "
                    f"with {best_model['Win_Rate']:.2%} win rate."
                )
            else:
                ai_insight = (
                    "🤖 Monthly AI Insight: No clear edge — performance varied across models."
                )

            monthly_insight_file = os.path.join(
                os.path.dirname(MONTHLY_SUMMARY_FILE), "monthly_ai_insight.txt"
            )
            with open(monthly_insight_file, "w") as f:
                f.write(ai_insight)
            logger.info(f"AI insight saved to {monthly_insight_file}")
            logger.info(ai_insight)
        except Exception as e:
            logger.warning(f"Failed to generate AI insight: {e}")

        # Save chart
        chart_file = os.path.join(
            os.path.dirname(MONTHLY_SUMMARY_FILE), "monthly_bankroll_comparison.png"
        )
        plt.title(f"Bankroll Trends for {current_month_str} by Model")
        plt.xlabel("Day Index")
        plt.ylabel("Final Bankroll")
        plt.legend()
        plt.grid(True)
        try:
            plt.savefig(chart_file)
            logger.info(f"📈 Chart saved to {chart_file}")
        except Exception as e:
            logger.error(f"Failed to save chart: {e}")
        plt.close()
    else:
        logger.warning(f"⚠️ No bankroll data found for month {current_month_str} to aggregate.")


if __name__ == "__main__":
    main()