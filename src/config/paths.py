# ============================================================
# Project: NBA Analytics & Betting Engine
# Author: Sadiq
# Description: Centralized filesystem paths for all pipeline data.
# ============================================================

from pathlib import Path

DATA_DIR = Path("data")

# Canonical snapshots
SCHEDULE_SNAPSHOT = DATA_DIR / "canonical" / "schedule.parquet"
LONG_SNAPSHOT = DATA_DIR / "canonical" / "long.parquet"

# Model registry
MODEL_REGISTRY_DIR = DATA_DIR / "models" / "registry"

# Predictions, odds, logs
PREDICTIONS_DIR = DATA_DIR / "predictions"
ODDS_DIR = DATA_DIR / "odds"
LOGS_DIR = DATA_DIR / "logs"

# Orchestrator logs
ORCHESTRATOR_LOG_DIR = DATA_DIR / "orchestrator_logs"

# Ensure all directories exist
for p in [
    DATA_DIR,
    MODEL_REGISTRY_DIR,
    PREDICTIONS_DIR,
    ODDS_DIR,
    LOGS_DIR,
    ORCHESTRATOR_LOG_DIR,
]:
    p.mkdir(parents=True, exist_ok=True)
