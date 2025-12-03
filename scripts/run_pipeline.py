# ============================================================
# File: scripts/run_pipeline.py
# Purpose: Fully automated NBA data pipeline with cleanup, archiving,
#          bankroll tracking, daily + weekly summaries, and logging
# ============================================================

import os
import subprocess
import shutil
import pandas as pd
from datetime import datetime
from core.config import (
    BASE_DATA_DIR, ARCHIVE_DIR, BASE_RESULTS_DIR, LOG_FILE,
    PICKS_BANKROLL_FILE, ensure_dirs, validate_config,
    DEFAULT_BANKROLL
)
from core.log_config import setup_logger
from core.exceptions import PipelineError

logger = setup_logger("pipeline")


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


def update_bankroll(picks_file: str):
    """Append today's picks results into bankroll tracking file."""
    if not os.path.exists(picks_file):
        logger.warning("⚠️ No picks.csv found, skipping bankroll update.")
        return

    df = pd.read_csv(picks_file)
    if df.empty:
        logger.info("ℹ️ Picks file is empty, skipping bankroll update.")
        return

    today = datetime.today().date().isoformat()
    total_stake = df["stake_amount"].sum()
    avg_ev = df["expected_value"].mean()
    bankroll_change = total_stake * avg_ev

    record = {
        "Date": today,
        "Total_Stake": total_stake,
        "Avg_EV": avg_ev,
        "Bankroll_Change": bankroll_change,
    }

    if os.path.exists(PICKS_BANKROLL_FILE):
        hist = pd.read_csv(PICKS_BANKROLL_FILE)
        hist = pd.concat([hist, pd.DataFrame([record])], ignore_index=True)
    else:
        hist = pd.DataFrame([record])

    hist.to_csv(PICKS_BANKROLL_FILE, index=False)
    logger.info(f"💰 Bankroll updated → {PICKS_BANKROLL_FILE}")


def log_daily_summary():
    """Log final bankroll, EV, and stake metrics from picks_bankroll.csv."""
    if not os.path.exists(PICKS_BANKROLL_FILE):
        logger.warning("⚠️ No picks_bankroll.csv found for summary logging.")
        return

    try:
        df = pd.read_csv(PICKS_BANKROLL_FILE)
        summary = df.tail(1).to_dict(orient="records")[0]

        # Compute cumulative bankroll
        cumulative = DEFAULT_BANKROLL + df["Bankroll_Change"].sum()

        logger.info("📊 Daily Summary:")
        logger.info(f"🏦 Final Bankroll: {cumulative:.2f}")
        logger.info(f"💰 Avg EV (today): {summary.get('Avg_EV', 'N/A')}")
        logger.info(f"🎯 Total Stake (today): {summary.get('Total_Stake', 'N/A')}")

        # Export summary to CSV
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
    if not os.path.exists(PICKS_BANKROLL_FILE):
        logger.warning("⚠️ No picks_bankroll.csv found for weekly summary.")
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


def run_step(cmd, step_name):
    """Run a subprocess step with logging and error handling."""
    logger.info(f"▶️ {step_name}...")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {step_name} failed: {e}")
        raise PipelineError(f"{step_name} failed: {e}")


def run_pipeline(skip_telegram: bool = False):
    logger.info("🚀 Starting automated NBA data pipeline")

    try:
        ensure_dirs()
        validate_config()

        # 1️⃣ Fetch season data (team stats)
        run_step(["python", "scripts/fetch_season_data.py"], "Fetching season data")

        # 2️⃣ Merge into SQLite database
        run_step(["python", "scripts/merge_team_data.py"], "Merging season data into SQLite")

        # 3️⃣ Cleanup: archive CSVs
        archive_csvs()

        # 4️⃣ Run prediction pipeline
        run_step(["python", "app/prediction_pipeline.py", "--model_type", "xgb", "--strategy", "kelly"],
                 "Running prediction pipeline")

        # 5️⃣ Update bankroll tracking
        update_bankroll(BASE_RESULTS_DIR / "picks.csv")

        # 6️⃣ Send Telegram report (optional)
        if not skip_telegram:
            run_step(["python", "scripts/telegram_report.py"], "Sending Telegram report")
        else:
            logger.info("📲 Skipping Telegram report (flag set)")

        # 7️⃣ Log daily summary
        log_daily_summary()

        # 8️⃣ Log weekly summary
        log_weekly_summary()

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