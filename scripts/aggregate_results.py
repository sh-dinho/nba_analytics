# ============================================================
# File: scripts/aggregate_results.py
# Purpose: Merge results from all model types into one summary CSV + chart
#          and append/overwrite centralized pipeline_summary.csv with season + notes
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from core.log_config import init_global_logger
from core.config import RESULTS_DIR, SUMMARY_FILE, DEFAULT_BANKROLL
from core.exceptions import DataError

logger = init_global_logger()

def main(export_json: bool = False, overwrite: bool = False, season: str = "aggregate", notes: str = ""):
    model_types = ["logistic", "xgb", "nn"]
    summaries = []

    plt.figure(figsize=(10, 6))

    for m in model_types:
        path = Path(RESULTS_DIR) / f"picks_bankroll_{m}.csv"
        if not path.exists():
            logger.warning(f"⚠️ Skipping {m}, no results file found.")
            continue

        try:
            df = pd.read_csv(path)
        except Exception as e:
            raise DataError(f"Failed to read {path}: {e}")

        df = df.rename(columns=str.lower)

        if df.empty:
            logger.warning(f"⚠️ Skipping {m}, file is empty.")
            continue

        final_bankroll = df["bankroll"].iloc[-1] if "bankroll" in df else DEFAULT_BANKROLL
        total_bets = len(df)
        wins = df["won"].sum() if "won" in df else 0
        win_rate = wins / total_bets if total_bets > 0 else 0
        avg_ev = df["ev"].mean() if "ev" in df else 0
        avg_stake = df["stake"].mean() if "stake" in df else 0

        summaries.append({
            "Model": m,
            "Final_Bankroll": round(final_bankroll, 2),
            "Total_Bets": total_bets,
            "Win_Rate": round(win_rate, 3),
            "Avg_EV": round(avg_ev, 3),
            "Avg_Stake": round(avg_stake, 2)
        })

        if "bankroll" in df:
            plt.plot(df.index, df["bankroll"], marker="o", label=f"{m} bankroll")

        if "won" in df:
            cum_wins = df["won"].cumsum()
            cum_win_rate = cum_wins / (df.index + 1)
            plt.plot(df.index, cum_win_rate * final_bankroll, linestyle="--", alpha=0.6,
                     label=f"{m} win-rate trend (scaled)")

    if summaries:
        summary_df = pd.DataFrame(summaries)
        os.makedirs(RESULTS_DIR, exist_ok=True)

        # Save combined summary
        combined_csv = Path(RESULTS_DIR) / "combined_summary.csv"
        summary_df.to_csv(combined_csv, index=False)
        logger.info(f"📊 Combined summary saved to {combined_csv}")

        if export_json:
            combined_json = Path(RESULTS_DIR) / "combined_summary.json"
            summary_df.to_json(combined_json, orient="records", indent=2)
            logger.info(f"📑 Combined summary also exported to {combined_json}")

        # Append or overwrite centralized pipeline_summary.csv
        run_time = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        summary_df["timestamp"] = run_time
        summary_df["season"] = season
        summary_df["target"] = "aggregate"
        summary_df["model_type"] = summary_df["Model"]
        summary_df["notes"] = notes

        if overwrite or not Path(SUMMARY_FILE).exists():
            summary_df.to_csv(SUMMARY_FILE, index=False)
            logger.info(f"📑 Centralized summary OVERWRITTEN at {SUMMARY_FILE}")
        else:
            summary_df.to_csv(SUMMARY_FILE, mode="a", header=False, index=False)
            logger.info(f"📑 Aggregated results appended to {SUMMARY_FILE}")

        # Save chart
        plt.title("Bankroll Trajectories by Model")
        plt.xlabel("Bet Number")
        plt.ylabel("Bankroll ($)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        chart_path = Path(RESULTS_DIR) / "bankroll_comparison.png"
        plt.savefig(chart_path)
        logger.info(f"📈 Bankroll comparison chart saved to {chart_path}")
    else:
        logger.error("❌ No results found to aggregate.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Aggregate model results into summary + chart")
    parser.add_argument("--export-json", action="store_true", help="Also export combined summary as JSON")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite centralized pipeline_summary.csv instead of appending")
    parser.add_argument("--season", type=str, default="aggregate", help="Season tag for aggregated entries (e.g. 2025-26)")
    parser.add_argument("--notes", type=str, default="", help="Optional notes to annotate aggregated entries")
    args = parser.parse_args()
    main(export_json=args.export_json, overwrite=args.overwrite, season=args.season, notes=args.notes)
