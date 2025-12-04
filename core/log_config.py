# ============================================================
# File: core/log_config.py
# Purpose: Centralized logger setup with UTF-8 encoding, rotation, and rich console output
# ============================================================

import logging
import sys
import io
from pathlib import Path
from logging.handlers import RotatingFileHandler
from rich.logging import RichHandler

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")
except Exception:
    pass

CONSOLE_FORMAT = "%(message)s"
FILE_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

def setup_logger(name: str,
                 log_file: Path | None = None,
                 level: int = logging.INFO,
                 max_bytes: int = 5_000_000,
                 backup_count: int = 5) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        console_handler = RichHandler(markup=True, rich_tracebacks=True)
        console_handler.setLevel(level)
        console_handler.setFormatter(logging.Formatter(CONSOLE_FORMAT))
        logger.addHandler(console_handler)

        if log_file:
            fh = RotatingFileHandler(log_file, maxBytes=max_bytes,
                                     backupCount=backup_count, encoding="utf-8")
            fh.setLevel(level)
            fh.setFormatter(logging.Formatter(FILE_FORMAT))
            logger.addHandler(fh)

    return logger

def init_global_logger(log_file: Path | None = None,
                       level: int = logging.INFO) -> logging.Logger:
    return setup_logger("nba_pipeline", log_file=log_file, level=level)
