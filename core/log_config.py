# ============================================================
# File: core/log_config.py
# Purpose: Global logging configuration for NBA Analytics
# ============================================================

import logging
import sys
from pathlib import Path
from core.paths import LOGS_DIR

LOG_FILE = LOGS_DIR / "pipeline.log"

def init_global_logger(name: str = "nba_analytics", log_to_file: bool = True) -> logging.Logger:
    """
    Initialize and return a global logger.

    Parameters
    ----------
    name : str
        Logger name (default: 'nba_analytics').
    log_to_file : bool
        Whether to write logs to a file in LOGS_DIR.

    Returns
    -------
    logging.Logger
        Configured logger.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        # Avoid adding multiple handlers if already configured
        return logger

    logger.setLevel(logging.INFO)

    # Formatter: [YYYY/MM/DD HH:MM:SS] LEVEL  message
    formatter = logging.Formatter(
        fmt='[%(asctime)s] %(levelname)-8s %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_to_file:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
