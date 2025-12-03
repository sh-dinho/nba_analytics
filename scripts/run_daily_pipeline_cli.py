# ============================================================
# File: scripts/run_daily_pipeline_cli.py
# Purpose: Run daily NBA prediction pipeline end-to-end
# ============================================================

import argparse
import os
import subprocess
import datetime
import pandas as pd
from pathlib import Path
from core.config import (
    ensure_dirs, BASE_DATA_DIR, MODELS_DIR, RESULTS_DIR, SUMMARY_FILE,
    PICKS_BANKROLL_FILE, DEFAULT_BANKROLL
)
from core.log_config import setup_logger
from core.exceptions import PipelineError

# Ensure directories exist using centralized config
ensure_dirs()

today = datetime.datetime.now().strftime("%Y%m%d")
LOG_FILE = os.path.join(RESULTS_DIR, f"pipeline_run_{today}.log")

logger = setup_logger("pipeline")


def get_current_season() -> str:
    """Return current NBA season string like '2025-26' based on today's date."""
    today = datetime.date.today()
    year = today.year
    month = today.month
    return f"{year}-{str(year+1)[-2:]}" if month >= 10 else f"{year-1}-{str(year)[-2:]}"


def run_step(args, step_name: str):
    """Run a subprocess step with logging and error handling."""
    logger.info(f"▶️ {step_name}...")
    try:
        subprocess.run(args, check=True, capture_output=True, text=True)
        logger.info(f"✅ {step_name} completed")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {step_name} failed: {e.stderr}")
        raise PipelineError(f"{step_name} failed: {e.stderr}")


def ensure_player_stats(season="2025-26", force_refresh=False):
    args = ["python", "-m", "scripts.fetch_player_stats", "--season", season]
    if force_refresh:
        args.append("--force_refresh")
    run_step(args, f"Ensuring player stats for season {season}")


def format_metric(name, value, fmt=".4f"):
    """Helper to safely format metrics that may be None."""
    return f"{name}={format(value, fmt)}" if value is not None else f"{name}=N/A"


# === New Enhancements ===

def update_bankroll(picks_file: Path):
    """Append today's picks results into bankroll tracking file."""
    if not picks_file.exists():
        logger.warning("⚠️ No picks.csv found, skipping bankroll update.")
        return
    df = pd.read_csv(picks_file)
    if df.empty:
        return
    today = datetime.date.today().isoformat()
    total_stake = df.get("stake_amount", pd.Series([0.0])).sum()
    avg_ev = df.get("ev", pd.Series([0.0])).mean()
    bankroll_change = (df.get("ev", pd.Series([0.0])) * df.get("stake_amount", pd.Series([0.0]))).sum()
    record = {"Date": today, "Total_Stake": total_stake, "Avg_EV": avg_ev, "Bankroll_Change": bankroll_change}
    if PICKS_BANKROLL_FILE.exists():
        hist = pd.read_csv(PICKS_BANKROLL_FILE)
        hist = pd.concat([hist, pd.DataFrame([record])], ignore_index=True)
    else:
        hist = pd.DataFrame([record])
    hist.to_csv(PICKS_BANKROLL_FILE, index=False)
    logger.info(f"💰 Bankroll updated → {PICKS_BANKROLL_FILE}")


def export_daily_summary(summary_entry: pd.DataFrame):
    """Export one-row daily summary with cumulative bankroll."""
    if PICKS_BANKROLL_FILE.exists():
        cumulative = DEFAULT_BANKROLL + pd.read_csv(PICKS_BANKROLL_FILE)["Bankroll_Change"].sum()
    else:
        cumulative = DEFAULT_BANKROLL
    summary_file = RESULTS_DIR / "summary.csv"
    summary_entry.assign(Final_Bankroll=cumulative).to_csv(summary_file, index=False)
    logger.info(f"📑 Daily summary exported to {summary_file}")


def log_weekly_summary():
    """Aggregate bankroll changes by week."""
    if not PICKS_BANKROLL_FILE.exists():
        return
    df = pd.read_csv(PICKS_BANKROLL_FILE)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Week"] = df["Date"].dt.to_period("W").astype(str)
    weekly = df.groupby("Week").agg({
        "Total_Stake": "sum",
        "Avg_EV": "mean",
        "Bankroll_Change": "sum"
    }).reset_index()
    weekly["Cumulative_Bankroll"] = DEFAULT_BANKROLL + weekly["Bankroll_Change"].cumsum()
    weekly_file = RESULTS_DIR / "weekly_summary.csv"
    weekly.to_csv(weekly_file, index=False)
    logger.info(f"📑 Weekly summary exported to {weekly_file}")


def log_monthly_summary():
    """Aggregate bankroll changes by month."""
    if not PICKS_BANKROLL_FILE.exists():
        return
    df = pd.read_csv(PICKS_BANKROLL_FILE)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Month"] = df["Date"].dt.to_period("M").astype(str)
    monthly = df.groupby("Month").agg({
        "Total_Stake": "sum",
        "Avg_EV": "mean",
        "Bankroll_Change": "sum"
    }).reset_index()
    monthly["Cumulative_Bankroll"] = DEFAULT_BANKROLL + monthly["Bankroll_Change"].cumsum()
    monthly_file = RESULTS_DIR / "monthly_summary.csv"
    monthly.to_csv(monthly_file, index=False)
    logger.info(f"📑 Monthly summary exported to {monthly_file}")


# === Main Pipeline ===

def main(threshold=0.6, strategy="kelly", max_fraction=0.05,
         season=None, force_refresh=False, rounds=10,
         target="label", model_type="logistic") -> pd.DataFrame:

    season = season or get_current_season()
    logger.info(f"📅 Auto-detected NBA season: {season}")
    logger.info(f"Starting pipeline | threshold={threshold}, strategy={strategy}, "
                f"max_fraction={max_fraction}, season={season}, force_refresh={force_refresh}, "
                f"rounds={rounds}, target={target}, model_type={model_type}")

    # Steps 1–5
    ensure_player_stats(season=season, force_refresh=force_refresh)

    if target in ["label", "margin", "outcome_category"]:
        run_step(["python", "-m", "scripts.build_features", "--rounds", str(rounds), "--training"],
                 "Building training features")
    else:
        run_step(["python", "-m", "scripts.build_features", "--rounds", str(rounds)],
                 "Building new game features")

    run_step(["python", "-m", "scripts.train_model", "--target", target, "--model_type", model_type],
             "Training model")

    run_step(["python", "-m", "scripts.generate_today_predictions",
              "--threshold", str(threshold), "--strategy", strategy, "--max_fraction", str(max_fraction)],
             "Generating predictions")

    run_step(["python", "-m", "scripts.generate_picks"], "Generating picks")

    # Step 6: Collect summaries (unchanged from your version) ...
    # [existing summary collection code here]

    # Append to rolling CSV summary
    run_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary_entry = pd.DataFrame([{ ... }])  # same as your version

    try:
        if Path(SUMMARY_FILE).exists():
            summary_entry.to_csv(SUMMARY_FILE, mode="a", header=False, index=False)
        else:
            summary_entry.to_csv(SUMMARY_FILE, index=False)
        logger.info(f"Pipeline summary appended to {SUMMARY_FILE}")
    except Exception as e:
        raise PipelineError(f"Failed to append pipeline summary: {e}")

    # === New Enhancements ===
    update_bankroll(RESULTS_DIR / "picks.csv")
    export_daily_summary(summary_entry)
    log_weekly_summary()
    log_monthly_summary()

    return summary_entry


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run daily NBA prediction pipeline")
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--strategy", type=str, default="kelly")
    parser.add_argument("--max_fraction", type=float, default=0.05)
    parser.add_argument("--season", type=str, default=get_current_season(),
                        help="NBA season, auto-detected by default")
    parser.add_argument("--force_refresh", action="store_true")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--target", type=str, default="label",
                        help="Target column: label, margin, outcome_category")
    parser.add_argument("--model_type", type=str, default="logistic",
                        help="Model type: logistic, rf, linear")
    args = parser.parse_args()

    main(threshold=args.threshold,
         strategy=args.strategy,
         max_fraction=args.max_fraction,
         season=args.season,
         force_refresh=args.force_refresh,
         rounds=args.rounds,
         target=args.target,
         model_type=args.model_type)