# ============================================================
# 🏀 NBA Analytics v3
# Module: Path Configuration
# File: src/config/paths.py
# Author: Sadiq
#
# Description:
#     Centralized file/directory paths for the entire project.
#     Ensures consistent structure across ingestion, features,
#     model training, predictions, backtesting, and dashboard.
# ============================================================

from __future__ import annotations

from pathlib import Path

# Root directory (auto-detect)
ROOT_DIR = Path(__file__).resolve().parents[2]

# Data directories
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Canonical snapshots
CANONICAL_DIR = DATA_DIR / "canonical"
CANONICAL_DIR.mkdir(exist_ok=True)

SCHEDULE_SNAPSHOT = CANONICAL_DIR / "schedule.parquet"
LONG_SNAPSHOT = CANONICAL_DIR / "long.parquet"

# Features
FEATURES_DIR = DATA_DIR / "features"
FEATURES_DIR.mkdir(exist_ok=True)

# Model registry
MODEL_REGISTRY_DIR = DATA_DIR / "models" / "registry"
MODEL_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)

# Predictions
PREDICTIONS_DIR = DATA_DIR / "predictions"
PREDICTIONS_DIR.mkdir(exist_ok=True)

# Odds
ODDS_DIR = DATA_DIR / "odds"
ODDS_DIR.mkdir(exist_ok=True)

# Reports
REPORTS_DIR = DATA_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# Orchestrator logs
ORCHESTRATOR_LOG_DIR = DATA_DIR / "orchestrator_logs"
ORCHESTRATOR_LOG_DIR.mkdir(exist_ok=True)

# Ensure all directories exist
for p in [
    DATA_DIR,
    MODEL_REGISTRY_DIR,
    PREDICTIONS_DIR,
    ODDS_DIR,
    ORCHESTRATOR_LOG_DIR,
]:
    p.mkdir(parents=True, exist_ok=True)
