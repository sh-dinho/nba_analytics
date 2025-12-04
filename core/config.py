# ============================================================
# File: core/config.py
# Purpose: Centralized configuration for NBA analytics pipeline
# ============================================================

import os
from pathlib import Path
from core.paths import (
    DATA_DIR, MODELS_DIR, RESULTS_DIR, ARCHIVE_DIR, LOGS_DIR,
    ensure_dirs, PLAYER_STATS_FILE, TRAINING_FEATURES_FILE
)
from core.log_config import init_global_logger
from core.exceptions import ConfigError

ENV = os.getenv("PIPELINE_ENV", "local").lower()
PROJECT_ROOT = Path(__file__).resolve().parent.parent

LOG_FILE = LOGS_DIR / "pipeline.log"
logger = init_global_logger(log_file=LOG_FILE)

# General settings
SEED = 42
DEFAULT_THRESHOLD = float(os.getenv("DEFAULT_THRESHOLD", "0.6"))
DEFAULT_BANKROLL = float(os.getenv("DEFAULT_BANKROLL", "1000.0"))
MAX_KELLY_FRACTION = float(os.getenv("MAX_KELLY_FRACTION", "0.05"))

PRINT_ONLY_ACTIONABLE = os.getenv("PRINT_ONLY_ACTIONABLE", "false").lower() in ("1", "true", "yes")
EV_THRESHOLD = float(os.getenv("EV_THRESHOLD", "0.01"))
MIN_KELLY_STAKE = float(os.getenv("MIN_KELLY_STAKE", "1.0"))

CLEANUP_MODE = os.getenv("CLEANUP_MODE", "archive").lower()
ARCHIVE_RETENTION_DAYS = int(os.getenv("ARCHIVE_RETENTION_DAYS", "180"))
MAX_DASHBOARD_IMAGES = int(os.getenv("MAX_DASHBOARD_IMAGES", "10"))
MAX_LOG_FILES = int(os.getenv("MAX_LOG_FILES", "20"))

USE_ROLLING_AVG = os.getenv("USE_ROLLING_AVG", "true").lower() in ("1", "true", "yes")
ROLLING_WINDOW = int(os.getenv("ROLLING_WINDOW", "5"))

def validate_config():
    """Warn on missing directories; CI/prod strictness handled in ensure_dirs(strict=True)."""
    issues = []

    if not DATA_DIR.exists():
        issues.append(f"Missing data directory: {DATA_DIR}")
    if not MODELS_DIR.exists():
        issues.append(f"Missing models directory: {MODELS_DIR}")
    if not RESULTS_DIR.exists():
        issues.append(f"Missing results directory: {RESULTS_DIR}")
    if not ARCHIVE_DIR.exists():
        issues.append(f"Missing archive directory: {ARCHIVE_DIR}")
    if not LOGS_DIR.exists():
        issues.append(f"Missing logs directory: {LOGS_DIR}")

    for issue in issues:
        logger.warning(issue)

    if not issues:
        logger.info("✅ Config validation passed — all directories exist.")

    mode = "rolling" if USE_ROLLING_AVG else "season"
    logger.info(f"📊 Feature building mode: {mode} averages (window={ROLLING_WINDOW if USE_ROLLING_AVG else 'N/A'})")
