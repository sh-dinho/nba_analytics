# ============================================================
# File: scripts/run_pipeline.py
# Purpose: Fully automated NBA data pipeline with cleanup, archiving,
#          bankroll tracking, daily + weekly + monthly summaries, and logging
# ============================================================

import os
import subprocess
import shutil
import pandas as pd
from datetime import datetime
from pathlib import Path
from core.config import (
    BASE_DATA_DIR, ARCHIVE_DIR, BASE_RESULTS_DIR,
    PICKS_BANKROLL_FILE, ensure_dirs, validate_config,
    DEFAULT_BANKROLL
)
from core.log_config import init_global_logger
from core.exceptions import PipelineError
from notifications import send_telegram_message  # ✅ optional

logger = init_global_logger()


def archive_csvs():
    """Move processed season CSVs into archive folder with timestamp."""
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    seasons_dir = os.path.join(BASE_DATA_DIR, "seasons")
    if not os.path.exists(seasons_dir):
        logger.warning("⚠️ No seasons folder found to archive.")
        return

    for file in os.listdir(seasons_dir):
        if file.endswith(".csv"):
            src = os.path.join(seasons_dir, file)
            dest_dir = os.path.join(ARCHIVE_DIR, timestamp)
            os.makedirs(dest_dir, exist_ok=True)
            dest = os.path.join(dest_dir, file)
            shutil.move(src, dest)
            logger.info(f"📦 Archived {file} → {dest}")


def update_bankroll(picks_file: Path):
    """Append today's picks results into bankroll tracking file."""
    if not picks_file.exists():
        logger.warning("⚠️ No picks.csv found, skipping bankroll update.")
        return

    df = pd.read_csv(picks_file)
    if df.empty:
        logger.info("ℹ️ Picks file is empty, skipping bankroll update.")
        return

    today = datetime.today().date().isoformat()
    total_stake = df.get("stake_amount", pd.Series([0.0])).sum()
    avg_ev = df.get("ev", pd.Series([0.0])).mean()
    bankroll_change = (df.get("ev", pd.Series([0.0])) * df.get("stake_amount", pd.Series([0.0]))).sum()

    record = {
        "Date": today,
        "Total_Stake": total_stake,
        "Avg_EV": avg_ev,
        "Bankroll_Change": bankroll_change,
    }

    if PICKS_BANKROLL_FILE.exists():
        hist = pd.read_csv(PICKS_BANKROLL_FILE)
        hist = pd.concat([hist, pd.DataFrame([record])], ignore_index=True)
    else:
        hist = pd.DataFrame([record])

    hist.to_csv(PICKS_BANKROLL_FILE, index=False)
    logger.info(f"💰 Bankroll updated → {PICKS_BANKROLL_FILE}")


def log_daily_summary():
    """Log final bankroll, EV, and stake metrics from picks_bankroll.csv."""
    if not PICKS_BANKROLL_FILE.exists():
        logger.warning("⚠️ No picks_bankroll.csv found for summary logging.")
        return

    try:
        df = pd.read_csv(PICKS_BANKROLL_FILE)
        summary = df.tail(1).to_dict(orient="records")[0]
        cumulative = DEFAULT_BANKROLL + df["Bankroll_Change"].sum()

        logger.info("📊 Daily Summary:")
        logger.info(f"🏦 Final Bankroll: {cumulative:.2f}")
        logger.info(f"💰 Avg EV (today): {summary.get('Avg_EV', 'N/A')}")
        logger.info(f"🎯 Total Stake (today): {summary.get('Total_Stake', 'N/A')}")

        summary_file = BASE_RESULTS_DIR / "summary.csv"
        pd.DataFrame([{
            "Date": summary["Date"],
            "Final_Bankroll": cumulative,
            "Avg_EV": summary.get("Avg_EV"),
            "Total_Stake": summary.get("Total_Stake"),
        }]).to_csv(summary_file, index=False)
        logger.info(f"📑 Daily summary exported to {summary_file}")

    except Exception as e:
        logger.error(f"❌ Failed to log daily summary: {e}")


def log_weekly_summary():
    """Aggregate bankroll changes by week for trend analysis."""
    if not PICKS_BANKROLL_FILE.exists():
        return
    try:
        df = pd.read_csv(PICKS_BANKROLL_FILE)
        df["Date"] = pd.to_datetime(df["Date"])
        df["Week"] = df["Date"].dt.to_period("W").astype(str)
        weekly = df.groupby("Week").agg({
            "Total_Stake": "sum",
            "Avg_EV": "mean",
            "Bankroll_Change": "sum"
        }).reset_index()
        weekly["Cumulative_Bankroll"] = DEFAULT_BANKROLL + weekly["Bankroll_Change"].cumsum()
        weekly_file = BASE_RESULTS_DIR / "weekly_summary.csv"
        weekly.to_csv(weekly_file, index=False)
        logger.info(f"📑 Weekly summary exported to {weekly_file}")
    except Exception as e:
        logger.error(f"❌ Failed to log weekly summary: {e}")


def log_monthly_summary():
    """Aggregate bankroll changes by month for trend analysis."""
    if not PICKS_BANKROLL_FILE.exists():
        return
    try:
        df = pd.read_csv(PICKS_BANKROLL_FILE)
        df["Date"] = pd.to_datetime(df["Date"])
        df["Month"] = df["Date"].dt.to_period("M").astype(str)
        monthly = df.groupby("Month").agg({
            "Total_Stake": "sum",
            "Avg_EV": "mean",
            "Bankroll_Change": "sum"
        }).reset_index()
        monthly["Cumulative_Bankroll"] = DEFAULT_BANKROLL + monthly["Bankroll_Change"].cumsum()
        monthly_file = BASE_RESULTS_DIR / "monthly_summary.csv"
        monthly.to_csv(monthly_file, index=False)
        logger.info(f"📑 Monthly summary exported to {monthly_file}")
    except Exception as e:
        logger.error(f"❌ Failed to log monthly summary: {e}")


def run_step(cmd, step_name):
    """Run a subprocess step with logging and error handling."""
    logger.info(f"▶️ {step_name}...")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"✅ {step_name} completed")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {step_name} failed: {e.stderr}")
        raise PipelineError(f"{step_name} failed: {e.stderr}")


def run_pipeline(skip_telegram: bool = False):
    logger.info("🚀 Starting automated NBA data pipeline")
    try:
        ensure_dirs()
        validate_config()

        run_step(["python", "scripts/fetch_season_data.py"], "Fetching season data")
        run_step(["python", "scripts/merge_team_data.py"], "Merging season data into SQLite")
        archive_csvs()
        run_step(["python", "app/prediction_pipeline.py", "--model_type", "xgb", "--strategy", "kelly"],
                 "Running prediction pipeline")

        update_bankroll(BASE_RESULTS_DIR / "picks.csv")

        if not skip_telegram:
            run_step(["python", "scripts/telegram_report.py"], "Sending Telegram report")
            # ✅ Direct notification
            send_telegram_message("🏀 Pipeline complete. Daily/weekly/monthly summaries updated.")
        else:
            logger.info("📲 Skipping Telegram report (flag set)")

        log_daily_summary()
        log_weekly_summary()
        log_monthly_summary()

        logger.info("✅ Pipeline completed successfully")
    except Exception as e:
        logger.error(f"❌ Unexpected pipeline error: {e}")
        raise PipelineError(f"Unexpected pipeline error: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run NBA analytics pipeline")
    parser.add_argument("--skip-telegram", action="store_true", help="Skip Telegram report step")
    args = parser.parse_args()
    run_pipeline(skip_telegram=args.skip_telegram)
