# ============================================================
# File: app/dashboard/archive_rotation.py
# Purpose: Archive old dashboard images and pipeline logs instead of deleting
# ============================================================

import os
import glob
import shutil
import datetime
from pathlib import Path
from core.config import RESULTS_DIR, ARCHIVE_DIR
from core.log_config import setup_logger

logger = setup_logger("archive_rotation")

# Keep only the last N files in results/, move older ones to archive
MAX_DASHBOARD_IMAGES = 10
MAX_LOG_FILES = 20

def archive_files(pattern: str, max_files: int, archive_subdir: str):
    """Move older files matching pattern into archive folder."""
    files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if len(files) > max_files:
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
    logger.info("🚀 Starting archive rotation")

    # Archive dashboards
    archive_files(os.path.join(RESULTS_DIR, "dashboard_*.png"), MAX_DASHBOARD_IMAGES, "dashboards")
    archive_files(os.path.join(RESULTS_DIR, "weekly_dashboard_*.png"), MAX_DASHBOARD_IMAGES, "dashboards")
    archive_files(os.path.join(RESULTS_DIR, "monthly_dashboard_*.png"), MAX_DASHBOARD_IMAGES, "dashboards")
    archive_files(os.path.join(RESULTS_DIR, "combined_dashboard_*.png"), MAX_DASHBOARD_IMAGES, "dashboards")

    # Archive pipeline logs
    archive_files(os.path.join(RESULTS_DIR, "pipeline_run_*.log"), MAX_LOG_FILES, "logs")

    logger.info("✅ Archive rotation complete")

if __name__ == "__main__":
    main()