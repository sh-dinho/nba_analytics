# ============================================================
# File: scripts/run_pipeline.py
# Purpose: Fully automated NBA data pipeline with cleanup, archiving, and logging
# ============================================================

import os
import subprocess
import shutil
import pandas as pd
from datetime import datetime
from core.config import BASE_DATA_DIR, ARCHIVE_DIR, LOG_DIR, LOG_FILE, PICKS_BANKROLL_FILE
from core.log_config import setup_logger
from core.exceptions import PipelineError

logger = setup_logger("pipeline")


def archive_csvs():
    """Move processed season CSVs into archive folder with timestamp."""
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    for file in os.listdir(os.path.join(BASE_DATA_DIR, "seasons")):
        if file.endswith(".csv"):
            src = os.path.join(BASE_DATA_DIR, "seasons", file)
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
    except Exception as e:
        logger.error(f"❌ Failed to log daily summary: {e}")


def run_pipeline():
    logger.info("🚀 Starting automated NBA data pipeline")

    try:
        # 1️⃣ Fetch season data (team stats)
        logger.info("📥 Fetching season data...")
        subprocess.run(["python", "scripts/fetch_season_data.py"], check=True)

        # 2️⃣ Merge into SQLite database
        logger.info("🗂️ Merging season data into SQLite...")
        subprocess.run(["python", "scripts/merge_team_data.py"], check=True)

        # 3️⃣ Cleanup: archive CSVs
        logger.info("🧹 Archiving processed CSVs...")
        archive_csvs()

        # 4️⃣ Run prediction pipeline
        logger.info("🤖 Running prediction pipeline...")
        subprocess.run([
            "python", "app/prediction_pipeline.py",
            "--model_type", "xgb", "--strategy", "kelly"
        ], check=True)

        # 5️⃣ Send Telegram report
        logger.info("📲 Sending Telegram report...")
        subprocess.run(["python", "scripts/telegram_report.py"], check=True)

        # 6️⃣ Log daily summary
        log_daily_summary()

        logger.info("✅ Pipeline completed successfully")

    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Pipeline step failed: {e}")
        raise PipelineError(f"Pipeline execution failed: {e}")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        raise PipelineError(f"Unexpected pipeline error: {e}")


if __name__ == "__main__":
    run_pipeline()