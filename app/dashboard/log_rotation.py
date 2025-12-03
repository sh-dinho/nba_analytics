# ============================================================
# File: app/dashboard/log_rotation.py
# Purpose: Rotate and clean up old dashboard images and pipeline logs
# ============================================================

import os
import glob
import datetime
from pathlib import Path
from core.config import RESULTS_DIR, LOG_FILE
from core.log_config import setup_logger

logger = setup_logger("log_rotation")

# Keep only the last N files per type
MAX_DASHBOARD_IMAGES = 10
MAX_LOG_FILES = 20

def rotate_files(pattern: str, max_files: int):
    """Keep only the most recent max_files matching pattern, delete older ones."""
    files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if len(files) > max_files:
        for old_file in files[max_files:]:
            try:
                os.remove(old_file)
                logger.info(f"🗑️ Deleted old file: {old_file}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to delete {old_file}: {e}")

def main():
    logger.info("🚀 Starting log rotation and cleanup")

    # Rotate dashboard images
    rotate_files(os.path.join(RESULTS_DIR, "dashboard_*.png"), MAX_DASHBOARD_IMAGES)
    rotate_files(os.path.join(RESULTS_DIR, "weekly_dashboard_*.png"), MAX_DASHBOARD_IMAGES)
    rotate_files(os.path.join(RESULTS_DIR, "monthly_dashboard_*.png"), MAX_DASHBOARD_IMAGES)
    rotate_files(os.path.join(RESULTS_DIR, "combined_dashboard_*.png"), MAX_DASHBOARD_IMAGES)

    # Rotate pipeline run logs
    rotate_files(os.path.join(RESULTS_DIR, "pipeline_run_*.log"), MAX_LOG_FILES)

    logger.info("✅ Log rotation complete")

if __name__ == "__main__":
    main()