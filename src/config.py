# ============================================================
# File: src/config.py
# Purpose: Central configuration for NBA Analytics v3
# Version: 3.0 (Unified + Canonical)
# Author: Your Team
# Date: December 2025
# ============================================================

from pathlib import Path

# ------------------------------------------------------------
# Base Directories
# ------------------------------------------------------------
BASE_DIR = Path(".").resolve()

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PARQUET_DIR = DATA_DIR / "parquet"
INGESTION_DIR = DATA_DIR / "ingestion"
FEATURES_DIR = DATA_DIR / "features"
PREDICTIONS_DIR = DATA_DIR / "predictions"
MODELS_DIR = DATA_DIR / "models"
REGISTRY_DIR = MODELS_DIR / "registry"

for d in [
    RAW_DIR,
    PARQUET_DIR,
    INGESTION_DIR,
    FEATURES_DIR,
    PREDICTIONS_DIR,
    MODELS_DIR,
    REGISTRY_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Ingestion
# ------------------------------------------------------------
SNAPSHOT_PATH = Path("data/ingestion/snapshot.parquet")
START_YEAR = 2022  # first season to pull from NBA API
USE_MOCK_DATA = False
# ------------------------------------------------------------
# Features
# ------------------------------------------------------------
ROLLING_WINDOW = 5  # games
FEATURES_DIR = FEATURES_DIR

# ------------------------------------------------------------
# Models
# ------------------------------------------------------------
MODEL_REGISTRY_DIR = REGISTRY_DIR
MODEL_PARAMS = {
    "n_estimators": 300,
    "max_depth": 8,
    "min_samples_split": 4,
    "min_samples_leaf": 2,
    "random_state": 42,
}

# ------------------------------------------------------------
# Predictions
# ------------------------------------------------------------
BATCH_PREDICTIONS_DIR = PREDICTIONS_DIR
LATEST_PREDICTIONS_PATH = PREDICTIONS_DIR / "predictions_latest.parquet"

# ------------------------------------------------------------
# Monitoring / Drift
# ------------------------------------------------------------
MONITORING = {
    "drift_alpha": 0.05,
    "min_samples": 20,
    "psi_buckets": 10,
}

# ------------------------------------------------------------
# Logging
# ------------------------------------------------------------
LOGGING = {
    "level": "INFO",
    "format": "<green>{time}</green> | <level>{level}</level> | {message}",
}
