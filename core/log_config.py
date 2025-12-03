# ============================================================
# File: core/log_config.py
# Purpose: Centralized logger setup with UTF-8 encoding
# ============================================================

import logging
import sys
import io
from pathlib import Path

# Force stdout/stderr to UTF-8 so emojis and Unicode characters work on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")


def setup_logger(name: str,
                 log_file: Path | None = None,
                 level: int = logging.INFO) -> logging.Logger:
    """
    Configure and return a logger with UTF-8 encoding.
    Logs to console and optionally to a file.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if setup_logger is called multiple times
    if not logger.handlers:
        # Console handler (UTF-8)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(logging.Formatter(
            "%(name)s - %(levelname)s - %(message)s"
        ))
        logger.addHandler(console_handler)

        # Optional file handler
        if log_file:
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setLevel(level)
            fh.setFormatter(logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            ))
            logger.addHandler(fh)

    return logger