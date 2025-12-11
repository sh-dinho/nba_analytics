# ============================================================
# File: src/utils/logging_config.py
# Purpose: Configure logging with console + daily rotating file handler
# Project: nba_analysis
# Version: 1.4 (adds log level override, validation, returns named logger)
# ============================================================

import logging
import os
from logging.handlers import TimedRotatingFileHandler


def configure_logging(
    log_file: str = None, retention_days: int = None, name: str = "nba_analysis"
) -> logging.Logger:
    """Configure logging with console + daily rotating file handler."""

    log_file = log_file or os.getenv("LOG_FILE", "logs/app.log")

    try:
        retention_days = int(retention_days or os.getenv("LOG_RETENTION_DAYS", "7"))
        if retention_days < 1:
            raise ValueError
    except ValueError:
        retention_days = 7
        logging.warning("Invalid LOG_RETENTION_DAYS, defaulting to 7")

    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    timed_handler = TimedRotatingFileHandler(
        log_file,
        when="midnight",
        interval=1,
        backupCount=retention_days,
        encoding="utf-8",
    )

    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, log_level, logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[timed_handler, logging.StreamHandler()],
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger(name)
    logger.setLevel(level)

    logger.info(
        "Logging configured. Writing to %s (retention: %d days, level: %s)",
        log_file,
        retention_days,
        log_level,
    )
    return logger
