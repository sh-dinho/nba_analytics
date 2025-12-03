# ============================================================
# File: scripts/weekly_aggregate.py
# Purpose: Aggregate 7-day results, track wins by model, add AI insight + chart
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from core.log_config import setup_logger
from core.exceptions import PipelineError

logger = setup_logger("weekly_aggregate")

RESULTS_DIR = "results"
SUMMARY_FILE = os.path.join(RESULTS_DIR, "weekly_summary.csv")
INSIGHT_FILE = os.path.join(RESULTS_DIR, "weekly_ai_insight.txt")
CHART_FILE = os.path.join(RESULTS_DIR, "weekly_bankroll_trends.png")


def main() -> pd.DataFrame | None:
    model_types = ["logistic", "xgb", "nn"]
    summaries = []

    cutoff = datetime.datetime.now() - datetime.timedelta(days=7)

    if not os.path.exists(RESULTS_DIR):
        raise PipelineError(f"Results directory not found: {RESULTS_DIR}")

    plt.figure(figsize=(10, 6))

    for m in model_types:
        files = [f for f in os.listdir(RESULTS_DIR) if f.startswith(f"picks_bankroll_{m}")]
        files.sort()

        total_wins, total_bets, final_bankrolls, daily_bankrolls = 0, 0, [], []

        for f in files:
            path = os.path.join(RESULTS_DIR, f)
            try:
                df = pd.read_csv(path)
            except Exception as e:
                logger.warning(f"⚠️ Skipping {path}: {e}")
                continue

            if "won" not in df.columns or "bankroll" not in df.columns:
                logger.warning(f"⚠️ Missing required columns in {path}. Skipping.")
                continue

            # Filter to last 7 days if Date column exists
            if "Date" in df.columns:
                try:
                    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                    df = df[df["Date"] >= cutoff]
                except Exception as e:
                    logger.warning(f"⚠️ Failed to parse Date in {path}: {e}")

            if df.empty:
                continue

            total_bets += len(df)
            total_wins += df["won"].loc[df["won"] != -1].sum()
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
            plt.plot(range(1, len(daily_bankrolls) + 1), daily_bankrolls, marker="o", label=m)

    if summaries:
        summary_df = pd.DataFrame(summaries)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        summary_df.to_csv(SUMMARY_FILE, index=False)
        logger.info(f"📊 Weekly summary saved to {SUMMARY_FILE}")

        # AI insight
        try:
            best_model = summary_df.loc[summary_df["Avg_Final_Bankroll"].idxmax()]
            if best_model["Win_Rate"] > 0.55:
                ai_insight = (
                    f"🤖 Weekly AI Insight: {best_model['Model']} shows sustained edge "
                    f"with {best_model['Win_Rate']:.2%} win rate."
                )
            else:
                ai_insight = (
                    "🤖 Weekly AI Insight: No clear edge — performance varied across models."
                )

            with open(INSIGHT_FILE, "w") as f:
                f.write(ai_insight)
            logger.info(f"AI insight saved to {INSIGHT_FILE}")
            logger.info(ai_insight)
        except Exception as e:
            logger.warning(f"⚠️ Failed to generate AI insight: {e}")

        # Save chart
        plt.title("Weekly Bankroll Trends by Model")
        plt.xlabel("Day Index")
        plt.ylabel("Final Bankroll")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        try:
            plt.savefig(CHART_FILE)
            logger.info(f"📈 Weekly bankroll chart saved to {CHART_FILE}")
        except Exception as e:
            logger.error(f"❌ Failed to save chart: {e}")
        plt.close()

        return summary_df
    else:
        logger.warning("❌ No weekly results found.")
        return None


if __name__ == "__main__":
    main()