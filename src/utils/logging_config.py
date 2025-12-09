# ============================================================
# Path: src/utils/logging_config.py
# Filename: logging_config.py
# Author: Your Team
# Date: December 2025
# Purpose: Configure logging for the NBA analytics project.
# ============================================================

import logging
import os

def configure_logging(log_file: str = "app.log"):
    """
    Configure logging to write both to a file and to the console.

    Args:
        log_file (str): Path to the log file.

    Returns:
        str: Path to the log file created.
    """
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)

    # Reset any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler()
        ]
    )

    logging.info("Logging configured. Writing to %s", log_file)
    return log_file
