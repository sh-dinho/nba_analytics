# ============================================================
# Path: tests/test_logging.py
# Filename: test_logging.py
# Author: Your Team
# Date: December 2025
# Purpose: Tests for logging configuration
# ============================================================

import os
import logging
from src.utils.logging_config import configure_logging

def test_configure_logging_creates_file_and_logs(tmp_path):
    # Use a temporary log file
    log_file = tmp_path / "test_app.log"

    # Configure logging
    configure_logging(str(log_file))

    # Write a test log entry
    logging.info("This is a test log entry")

    # Ensure file exists
    assert log_file.exists()

    # Ensure file is not empty
    content = log_file.read_text().strip()
    assert len(content) > 0

    # Ensure our test log entry is present
    assert "This is a test log entry" in content
