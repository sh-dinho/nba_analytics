# ============================================================
# File: src/utils/logging.py
# Purpose: Central logging configuration for NBA analysis project
# Project: nba_analysis
# ============================================================

import os
import sys
from loguru import logger

# Track whether logging has already been configured
_LOGGING_CONFIGURED = False


def configure_logging(
    level: str = "INFO",
    log_dir: str = "logs",
    name: str = "nba_analysis",
    rotation: str = "10 MB",
    retention: str = "7 days",
):
    """Configure Loguru logging with file + stdout sinks and contextual binding."""
    global _LOGGING_CONFIGURED

    if not _LOGGING_CONFIGURED:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{name}.log")

        fmt = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}"

        # Reset previous handlers
        logger.remove()

        # File sink
        logger.add(
            log_path,
            rotation=rotation,
            retention=retention,
            level=level,
            format=fmt,
            enqueue=True,  # safer for multi-process logging
        )

        # Console sink
        logger.add(sys.stdout, level=level, format=fmt)

        # Catch uncaught exceptions
        def exception_handler(exc_type, exc_value, exc_traceback):
            logger.opt(exception=True).error(
                "Uncaught exception:", exc_info=(exc_type, exc_value, exc_traceback)
            )

        sys.excepthook = exception_handler

        _LOGGING_CONFIGURED = True

    # Return a contextual logger bound with module name
    return logger.bind(name=name)


def get_logger(name: str) -> logger:
    """
    Retrieve a contextual logger for a given module name.
    Ensures logging is configured once globally.
    """
    return configure_logging(name=name)
