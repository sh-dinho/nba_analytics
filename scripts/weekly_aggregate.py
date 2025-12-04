# ============================================================
# File: scripts/weekly_aggregate.py
# Purpose: Aggregate 7-day results, track wins by model, add AI insight + chart
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from pathlib import Path
import argparse
from core.log_config import init_global_logger
from core.exceptions import PipelineError
from core.config import SUMMARY_FILE as PIPELINE_SUMMARY_FILE

logger = init_global_logger()

RESULTS_DIR = "results"
SUMMARY_FILE = Path(RESULTS_DIR) / "weekly_summary.csv"
INSIGHT_FILE = Path(RESULTS_DIR) / "weekly_ai_insight.txt"
CHART_FILE = Path(RESULTS_DIR) / "weekly_bankroll_trends.png"


def main(export_json: bool = False,
         append_pipeline: bool = True,
         overwrite: bool = False,
         season: str = "aggregate",
         notes: str = "weekly aggregate") -> pd.DataFrame | None:

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

        if export_json:
            summary_df.to_json(Path(RESULTS_DIR) / "weekly_summary.json", orient="records", indent=2)
            logger.info("📑 Weekly summary also exported to JSON")

        # Append or overwrite centralized pipeline_summary.csv
        if append_pipeline:
            run_time = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            summary_df["timestamp"] = run_time
            summary_df["season"] = season
            summary_df["target"] = "aggregate"
            summary_df["model_type"] = summary_df["Model"]
            summary_df["notes"] = notes

            if overwrite or not Path(PIPELINE_SUMMARY_FILE).exists():
                summary_df.to_csv(PIPELINE_SUMMARY_FILE, index=False)
                logger.info(f"📑 Centralized summary OVERWRITTEN at {PIPELINE_SUMMARY_FILE}")
            else:
                summary_df.to_csv(PIPELINE_SUMMARY_FILE, mode="a", header=False, index=False)
                logger.info(f"📑 Weekly results appended to {PIPELINE_SUMMARY_FILE}")

        # AI insight
        try:
            best_model = summary_df.loc[summary_df["Avg_Final_Bankroll"].idxmax()]
            if best_model["Win_Rate"] > 0.55:
                ai_insight = (
                    f"🤖 Weekly AI Insight: {best_model['Model']} shows sustained edge "
                    f"with {best_model['Win_Rate']:.2%} win rate."
                )
            else:
                ai_insight = "🤖 Weekly AI Insight: No clear edge — performance varied across models."

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
    parser = argparse.ArgumentParser(description="Aggregate 7-day results and generate weekly summary + chart")
    parser.add_argument("--export-json", action="store_true", help="Also export weekly summary as JSON")
    parser.add_argument("--no-append", action="store_true", help="Do not append to centralized pipeline_summary.csv")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite centralized pipeline_summary.csv instead of appending")
    parser.add_argument("--season", type=str, default="aggregate", help="Season tag for weekly entries (e.g. 2025-26)")
    parser.add_argument("--notes", type=str, default="weekly aggregate", help="Optional notes to annotate weekly entries")
    args = parser.parse_args()

    main(export_json=args.export_json,
         append_pipeline=not args.no_append,
         overwrite=args.overwrite,
         season=args.season,
         notes=args.notes)
