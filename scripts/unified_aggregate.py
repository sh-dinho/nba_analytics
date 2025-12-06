# ============================================================
# File: scripts/unified_aggregate.py
# Purpose: Aggregate results, generate monthly summary & chart, update AI tracker, send Telegram
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from pathlib import Path
import argparse

from nba_core.config import BASE_RESULTS_DIR, MONTHLY_SUMMARY_FILE, SUMMARY_FILE as PIPELINE_SUMMARY_FILE
from nba_core.log_config import init_global_logger
from nba_core.exceptions import PipelineError
from scripts.ai_tracker import update_tracker, plot_team_dashboard
from nba_core.paths import AI_TRACKER_TEAMS_FILE
from notifications import send_photo

logger = init_global_logger("unified_aggregate")
MODEL_TYPES = ["logistic", "xgb", "nn"]
CHART_FILE = Path(MONTHLY_SUMMARY_FILE).parent / "monthly_bankroll_comparison.png"

def main(export_json: bool = False,
         append_pipeline: bool = True,
         overwrite: bool = False,
         season: str = "aggregate",
         notes: str = "monthly aggregate",
         notify: bool = False) -> pd.DataFrame | None:

    if not BASE_RESULTS_DIR.exists():
        raise PipelineError(f"Results directory not found: {BASE_RESULTS_DIR}")

    summaries = []
    now = datetime.datetime.now()
    current_month_str = now.strftime("%Y-%m")

    plt.figure(figsize=(10, 6))

    for m in MODEL_TYPES:
        bankroll_files = sorted(f for f in os.listdir(BASE_RESULTS_DIR)
                                if f.startswith(f"picks_bankroll_{m}") and f.endswith(".csv"))

        total_wins, total_bets = 0, 0
        final_bankrolls_daily = []

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

            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                df_month = df[df["Date"].dt.strftime("%Y-%m") == current_month_str]
            else:
                df_month = df if current_month_str in f else pd.DataFrame()

            if not df_month.empty and {"won", "bankroll"}.issubset(df_month.columns):
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
        MONTHLY_SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(MONTHLY_SUMMARY_FILE, index=False)

        if export_json:
            summary_df.to_json(MONTHLY_SUMMARY_FILE.parent / "monthly_summary.json",
                               orient="records", indent=2)

        if append_pipeline:
            run_time = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            summary_df["timestamp"] = run_time
            summary_df["season"] = season
            summary_df["target"] = "aggregate"
            summary_df["model_type"] = summary_df["Model"]
            summary_df["notes"] = notes

            if overwrite or not Path(PIPELINE_SUMMARY_FILE).exists():
                summary_df.to_csv(PIPELINE_SUMMARY_FILE, index=False)
            else:
                summary_df.to_csv(PIPELINE_SUMMARY_FILE, mode="a", header=False, index=False)

        plt.title(f"Bankroll Trends for {current_month_str} by Model")
        plt.xlabel("Day Index")
        plt.ylabel("Final Bankroll")
        plt.legend()
        plt.grid(True)
        try:
            plt.savefig(CHART_FILE)
            if notify:
                try:
                    send_photo(str(CHART_FILE), caption=f"📈 Monthly Bankroll Trends ({current_month_str})")
                except Exception as e:
                    logger.warning(f"⚠️ Telegram chart send failed: {e}")
        except Exception as e:
            logger.error(f"Failed to save chart: {e}")
        plt.close()

        backtest_file = BASE_RESULTS_DIR / "unified_aggregate_results.csv"
        try:
            update_tracker(backtest_file, season=season, notes=notes, notify=notify)
        except Exception as e:
            logger.warning(f"⚠️ AI tracker update failed: {e}")

        if AI_TRACKER_TEAMS_FILE.exists():
            try:
                team_stats_df = pd.read_csv(AI_TRACKER_TEAMS_FILE)
                dashboard_img = plot_team_dashboard(team_stats_df, season=season)
                send_photo(str(dashboard_img), caption="📊 AI Tracker Team Win Rates")
            except Exception as e:
                logger.warning(f"⚠️ AI tracker dashboard failed: {e}")

        return summary_df

    else:
        logger.warning(f"No bankroll data found for month {current_month_str} to aggregate.")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified aggregation with AI tracker")
    parser.add_argument("--export-json", action="store_true")
    parser.add_argument("--no-append", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--season", type=str, default="aggregate")
    parser.add_argument("--notes", type=str, default="monthly aggregate")
    parser.add_argument("--notify", action="store_true")
    args = parser.parse_args()

    main(export_json=args.export_json,
         append_pipeline=not args.no_append,
         overwrite=args.overwrite,
         season=args.season,
         notes=args.notes,
         notify=args.notify)
