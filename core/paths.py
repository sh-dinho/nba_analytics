# ============================================================
# File: core/paths.py
# Purpose: Centralized path definitions and directory setup for NBA analytics pipeline
# ============================================================

from pathlib import Path
import os
from core.log_config import init_global_logger
from core.exceptions import ConfigError

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
ARCHIVE_DIR = DATA_DIR / "archive"
LOGS_DIR = PROJECT_ROOT / "logs"

# Common files
PLAYER_STATS_FILE = DATA_DIR / "player_stats.csv"
TRAINING_FEATURES_FILE = DATA_DIR / "training_features.csv"

logger = init_global_logger(log_file=LOGS_DIR / "pipeline.log")

def ensure_dirs(strict: bool = False):
    """Ensure all required directories exist. In strict mode, raise ConfigError in CI/prod instead of creating."""
    required = [DATA_DIR, MODELS_DIR, RESULTS_DIR, ARCHIVE_DIR, LOGS_DIR]
    for d in required:
        if not d.exists():
            if strict and os.getenv("PIPELINE_ENV") in ("ci", "prod"):
                raise ConfigError(f"Missing required directory: {d}")
            d.mkdir(parents=True, exist_ok=True)
            logger.info(f"📂 Created directory: {d}")
        else:
            logger.info(f"📂 Directory already exists: {d}")
