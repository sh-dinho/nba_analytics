# ============================================================
# File: src/utils/logging_config.py
# Purpose: Configure logging with console + daily rotating file handler
# Project: nba_analysis
# Version: 1.3 (adds environment variable overrides for log file and retention)
#
# Dependencies:
# - os (standard library)
# - logging (standard library)
# ============================================================

import logging
import os
from logging.handlers import TimedRotatingFileHandler


def configure_logging(
    log_file: str = None,
    retention_days: int = None
):
    """
    Configure logging with both console and daily rotating file handler.

    Arguments:
    log_file -- Path to the log file (default: logs/app.log or LOG_FILE env var)
    retention_days -- Number of days to retain logs (default: 7 or LOG_RETENTION_DAYS env var)
    """

    # Environment variable overrides
    log_file = log_file or os.getenv("LOG_FILE", "logs/app.log")
    retention_days = retention_days or int(os.getenv("LOG_RETENTION_DAYS", "7"))

    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)

    # Reset any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Configure timed rotating file handler (rotate at midnight, keep retention_days backups)
    timed_handler = TimedRotatingFileHandler(
        log_file, when="midnight", interval=1, backupCount=retention_days, encoding="utf-8"
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[timed_handler, logging.StreamHandler()],
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.info("Logging configured. Writing to %s (retention: %d days)", log_file, retention_days)
