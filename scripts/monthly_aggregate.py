# ============================================================
# File: scripts/monthly_aggregate.py
# Purpose: Aggregate monthly results, track wins by model, add AI insight + chart
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import argparse
from pathlib import Path
from core.config import BASE_RESULTS_DIR, MONTHLY_SUMMARY_FILE, SUMMARY_FILE as PIPELINE_SUMMARY_FILE
from core.log_config import init_global_logger
from core.exceptions import PipelineError
from notifications import send_message, send_photo

logger = init_global_logger()

MODEL_TYPES = ["logistic", "xgb", "nn"]
INSIGHT_FILE = Path(MONTHLY_SUMMARY_FILE).parent / "monthly_ai_insight.txt"
CHART_FILE = Path(MONTHLY_SUMMARY_FILE).parent / "monthly_bankroll_comparison.png"


def main(export_json: bool = False,
         append_pipeline: bool = True,
         overwrite: bool = False,
         season: str = "aggregate",
         notes: str = "monthly aggregate",
         notify: bool = False) -> pd.DataFrame | None:
    """Aggregate monthly results, track wins by model, add AI insight + chart."""
    summaries = []

    now = datetime.datetime.now()
    current_month_str = now.strftime("%Y-%m")

    if not BASE_RESULTS_DIR.exists():
        raise PipelineError(f"Results directory not found: {BASE_RESULTS_DIR}")

    plt.figure(figsize=(10, 6))

    for m in MODEL_TYPES:
        try:
            files_in_dir = os.listdir(BASE_RESULTS_DIR)
        except Exception as e:
            raise PipelineError(f"Failed to list results directory {BASE_RESULTS_DIR}: {e}")

        bankroll_files = [
            f for f in files_in_dir
            if f.startswith(f"picks_bankroll_{m}") and f.endswith(".csv")
        ]
        bankroll_files.sort()

        total_wins, total_bets = 0, 0
        final_bankrolls_daily = []

        logger.info(f"Processing {m} model with {len(bankroll_files)} files...")

        for f in bankroll_files:
            path = BASE_RESULTS_DIR / f
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
                    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                    df_month = df[df["Date"].dt.strftime("%Y-%m") == current_month_str]
                except Exception as e:
                    logger.warning(f"Failed to parse Date column in {f}: {e}")
            else:
                if current_month_str in f:
                    df_month = df

            if not df_month.empty:
                if "won" not in df_month.columns or "bankroll" not in df_month.columns:
                    logger.warning(f"Missing required columns in {f}. Skipping.")
                    continue

                total_bets += len(df_month)
                total_wins += df_month["won"].loc[df_month["won"] != -1].sum()
                final_bankrolls_daily.append(df_month.iloc[-1]["bankroll"])

        if total_bets > 0:
            avg_final_bankroll = sum(final_bankrolls_daily) / len(final_bankrolls_daily)
            summaries.append({
                "Model": m,
                "Total_Bets": total_bets,
                "Total_Wins": total_wins,
                "Win_Rate": total_wins / total_bets,
                "Avg_Final_Bankroll": avg_final_bankroll
            })
            plt.plot(range(1, len(final_bankrolls_daily) + 1),
                     final_bankrolls_daily, marker="o", label=m)

    if summaries:
        summary_df = pd.DataFrame(summaries)

        try:
            os.makedirs(Path(MONTHLY_SUMMARY_FILE).parent, exist_ok=True)
            summary_df.to_csv(MONTHLY_SUMMARY_FILE, index=False)
            logger.info(f"📊 Monthly summary saved to {MONTHLY_SUMMARY_FILE}")
        except Exception as e:
            raise PipelineError(f"Failed to save monthly summary: {e}")

        if export_json:
            summary_df.to_json(Path(MONTHLY_SUMMARY_FILE).parent / "monthly_summary.json",
                               orient="records", indent=2)
            logger.info("📑 Monthly summary also exported to JSON")

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
                logger.info(f"📑 Monthly results appended to {PIPELINE_SUMMARY_FILE}")

        # AI insight
        try:
            best_model = summary_df.loc[summary_df["Avg_Final_Bankroll"].idxmax()]
            if best_model["Win_Rate"] > 0.55:
                ai_insight = (
                    f"🤖 Monthly AI Insight: {best_model['Model']} shows sustained edge "
                    f"with {best_model['Win_Rate']:.2%} win rate."
                )
            else:
                ai_insight = "🤖 Monthly AI Insight: No clear edge — performance varied across models."

            with open(INSIGHT_FILE, "w") as f:
                f.write(ai_insight)
            logger.info(f"AI insight saved to {INSIGHT_FILE}")
            logger.info(ai_insight)

            if notify:
                send_message(ai_insight)
                logger.info("📲 AI insight pushed to Telegram")
        except Exception as e:
            logger.warning(f"Failed to generate AI insight: {e}")

        # Save chart
        plt.title(f"Bankroll Trends for {current_month_str} by Model")
        plt.xlabel("Day Index")
        plt.ylabel("Final Bankroll")
        plt.legend()
        plt.grid(True)
        try:
            plt.savefig(CHART_FILE)
            logger.info(f"📈 Chart saved to {CHART_FILE}")
            if notify:
                send_photo(str(CHART_FILE), caption=f"📈 Monthly Bankroll Trends ({current_month_str})")
                logger.info("📲 Monthly chart pushed to Telegram")
        except Exception as e:
            logger.error(f"Failed to save chart: {e}")
        plt.close()

        return summary_df
    else:
        logger.warning(f"⚠️ No bankroll data found for month {current_month_str} to aggregate.")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate monthly results and generate summary + chart")
    parser.add_argument("--export-json", action="store_true", help="Also export monthly summary as JSON")
    parser.add_argument("--no-append", action="store_true", help="Do not append to centralized pipeline_summary.csv")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite centralized pipeline_summary.csv instead of appending")
    parser.add_argument("--season", type=str, default="aggregate", help="Season tag for monthly entries (e.g. 2025-26)")
    parser.add_argument("--notes", type=str, default="monthly aggregate", help="Optional notes to annotate monthly entries")
    parser.add_argument("--notify", action="store_true", help="Send AI insight and chart to Telegram")
    args = parser.parse_args()

    main(export_json=args.export_json,
         append_pipeline=not args.no_append,
         overwrite=args.overwrite,
         season=args.season,
         notes=args.notes,
         notify=args.notify)
