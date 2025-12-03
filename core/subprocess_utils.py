# ============================================================
# File: core/subprocess_utils.py
# Purpose: Retry wrapper for subprocess calls
# ============================================================

import subprocess
import time
from core.log_config import setup_logger
from core.exceptions import PipelineError

logger = setup_logger("subprocess_utils")

def run_with_retry(cmd, retries=3, delay=5):
    for attempt in range(1, retries+1):
        try:
            subprocess.run(cmd, check=True)
            return
        except subprocess.CalledProcessError as e:
            logger.warning(f"Attempt {attempt} failed: {e}")
            if attempt == retries:
                raise PipelineError(f"Command failed after {retries} attempts: {cmd}")
            time.sleep(delay)