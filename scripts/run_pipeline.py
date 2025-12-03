# ============================================================
# File: scripts/run_pipeline.py
# Purpose: Fully automated NBA data pipeline with cleanup, archiving, and logging
# ============================================================

import os
import subprocess
import shutil
import pandas as pd
from datetime import datetime
from core.config import BASE_DATA_DIR, ARCHIVE_DIR, BASE_RESULTS_DIR, LOG_FILE, PICKS_BANKROLL_FILE, ensure_dirs, validate_config
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


def log_daily_summary():
    """Log final bankroll, win rate, EV, and Kelly metrics from picks_bankroll.csv."""
    if not os.path.exists(PICKS_BANKROLL_FILE):
        logger.warning("⚠️ No picks_bankroll.csv found for summary logging.")
        return

    try:
        df = pd.read_csv(PICKS_BANKROLL_FILE)
        summary = df.tail(1).to_dict(orient="records")[0]

        logger.info("📊 Daily Summary:")
        logger.info(f"🏦 Final Bankroll: {summary.get('Final_Bankroll', 'N/A')}")
        logger.info(f"✅ Win Rate: {summary.get('Win_Rate', 'N/A')}")
        logger.info(f"💰 Avg EV: {summary.get('Avg_EV', 'N/A')}")
        logger.info(f"🎯 Avg Kelly Bet: {summary.get('Avg_Kelly_Bet', 'N/A')}")

        # Export summary to CSV
        summary_file = BASE_RESULTS_DIR / "summary.csv"
        df.tail(1).to_csv(summary_file, index=False)
        logger.info(f"📑 Daily summary exported to {summary_file}")

    except Exception as e:
        logger.error(f"❌ Failed to log daily summary: {e}")


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

        # 5️⃣ Send Telegram report (optional)
        if not skip_telegram:
            run_step(["python", "scripts/telegram_report.py"], "Sending Telegram report")
        else:
            logger.info("📲 Skipping Telegram report (flag set)")

        # 6️⃣ Log daily summary
        log_daily_summary()

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