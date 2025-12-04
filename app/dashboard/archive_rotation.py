# ============================================================
# File: app/dashboard/archive_rotation.py
# Purpose: Archive old daily dashboard images and pipeline logs
# ============================================================

import os
import glob
import shutil
import datetime
from pathlib import Path
from core.config import RESULTS_DIR, ARCHIVE_DIR
from core.log_config import setup_logger

logger = setup_logger("archive_rotation")

# Maximum number of files to keep in results/
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

    # Daily Dashboard Images and CSVs
    dashboard_dir = RESULTS_DIR / "dashboard"
    archive_files(str(dashboard_dir / "bankroll_*.png"), MAX_DASHBOARD_IMAGES, "dashboards")
    archive_files(str(dashboard_dir / "bankroll_*.csv"), MAX_DASHBOARD_IMAGES, "dashboards")

    # Pipeline Logs
    archive_files(os.path.join(RESULTS_DIR, "pipeline_run_*.log"), MAX_LOG_FILES, "logs")

    logger.info("✅ Archive rotation complete")

if __name__ == "__main__":
    main()
