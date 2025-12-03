# ============================================================
# File: app/dashboard/cleanup.py
# Purpose: Cleanup old dashboards/logs (delete or archive based on config)
# ============================================================

import os
import glob
import shutil
import datetime
from pathlib import Path
from core.config import RESULTS_DIR, ARCHIVE_DIR, CLEANUP_MODE, MAX_DASHBOARD_IMAGES, MAX_LOG_FILES
from core.log_config import setup_logger

logger = setup_logger("cleanup")

def handle_files(pattern: str, max_files: int, archive_subdir: str):
    """Delete or archive files depending on CLEANUP_MODE."""
    files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if len(files) <= max_files:
        return

    if CLEANUP_MODE == "delete":
        for old_file in files[max_files:]:
            try:
                os.remove(old_file)
                logger.info(f"🗑️ Deleted old file: {old_file}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to delete {old_file}: {e}")

    elif CLEANUP_MODE == "archive":
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dest_dir = Path(ARCHIVE_DIR) / archive_subdir / timestamp
        dest_dir.mkdir(parents=True, exist_ok=True)
        for old_file in files[max_files:]:
            try:
                shutil.move(old_file, dest_dir / os.path.basename(old_file))
                logger.info(f"📦 Archived old file: {old_file} → {dest_dir}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to archive {old_file}: {e}")

def main():
    logger.info(f"🚀 Starting cleanup in mode: {CLEANUP_MODE}")

    # Dashboards
    handle_files(os.path.join(RESULTS_DIR, "dashboard_*.png"), MAX_DASHBOARD_IMAGES, "dashboards")
    handle_files(os.path.join(RESULTS_DIR, "weekly_dashboard_*.png"), MAX_DASHBOARD_IMAGES, "dashboards")
    handle_files(os.path.join(RESULTS_DIR, "monthly_dashboard_*.png"), MAX_DASHBOARD_IMAGES, "dashboards")
    handle_files(os.path.join(RESULTS_DIR, "combined_dashboard_*.png"), MAX_DASHBOARD_IMAGES, "dashboards")

    # Logs
    handle_files(os.path.join(RESULTS_DIR, "pipeline_run_*.log"), MAX_LOG_FILES, "logs")

    logger.info("✅ Cleanup complete")

if __name__ == "__main__":
    main()