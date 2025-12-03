# ============================================================
# File: core/config.py
# Purpose: Centralized configuration for NBA analytics pipeline
# ============================================================

from pathlib import Path
import os
import logging

# Environment setup
ENV = os.getenv("PIPELINE_ENV", "local").lower()
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if ENV in ("ci", "prod"):
    BASE_DATA_DIR = PROJECT_ROOT / "data"
    BASE_MODELS_DIR = PROJECT_ROOT / "models"
    BASE_RESULTS_DIR = PROJECT_ROOT / "results"
    BASE_LOGS_DIR = PROJECT_ROOT / "logs"
    DB_PATH = PROJECT_ROOT / "Data" / "TeamData.sqlite"
else:
    BASE_DIR = Path.home() / "nba_analytics"
    BASE_DATA_DIR = BASE_DIR / "data"
    BASE_MODELS_DIR = BASE_DIR / "models"
    BASE_RESULTS_DIR = BASE_DIR / "results"
    BASE_LOGS_DIR = BASE_DIR / "logs"
    DB_PATH = BASE_DIR / "Data" / "TeamData.sqlite"

# Backward compatibility aliases
RESULTS_DIR = BASE_RESULTS_DIR
MODELS_DIR = BASE_MODELS_DIR

# Extra pipeline paths
ARCHIVE_DIR = BASE_DATA_DIR / "archive"
LOG_FILE = BASE_LOGS_DIR / "pipeline.log"
PICKS_BANKROLL_FILE = BASE_RESULTS_DIR / "picks_bankroll_xgb.csv"

# NEW: Logs for downloads and training
DOWNLOAD_SUMMARY_FILE = BASE_LOGS_DIR / "download_summary.log"
TRAINING_METRICS_LOG = BASE_LOGS_DIR / "training_metrics.log"

def ensure_dirs():
    """Ensure all required directories exist, including NN_Models and XGBoost subfolders."""
    for d in [BASE_DATA_DIR, BASE_MODELS_DIR, BASE_RESULTS_DIR, BASE_LOGS_DIR, ARCHIVE_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    (BASE_MODELS_DIR / "NN_Models").mkdir(parents=True, exist_ok=True)
    (BASE_MODELS_DIR / "XGBoost").mkdir(parents=True, exist_ok=True)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

def dump_config():
    return {
        "ENV": ENV,
        "BASE_DATA_DIR": str(BASE_DATA_DIR),
        "BASE_MODELS_DIR": str(BASE_MODELS_DIR),
        "BASE_RESULTS_DIR": str(BASE_RESULTS_DIR),
        "BASE_LOGS_DIR": str(BASE_LOGS_DIR),
        "DB_PATH": str(DB_PATH),
        "ARCHIVE_DIR": str(ARCHIVE_DIR),
        "LOG_FILE": str(LOG_FILE),
        "PICKS_BANKROLL_FILE": str(PICKS_BANKROLL_FILE),
        "DOWNLOAD_SUMMARY_FILE": str(DOWNLOAD_SUMMARY_FILE),
        "TRAINING_METRICS_LOG": str(TRAINING_METRICS_LOG),
        "SEED": SEED,
        "DEFAULT_THRESHOLD": DEFAULT_THRESHOLD,
        "DEFAULT_BANKROLL": DEFAULT_BANKROLL,
        "MAX_KELLY_FRACTION": MAX_KELLY_FRACTION,
        "CLEANUP_MODE": CLEANUP_MODE,
        "ARCHIVE_RETENTION_DAYS": ARCHIVE_RETENTION_DAYS,
        "MAX_DASHBOARD_IMAGES": MAX_DASHBOARD_IMAGES,
        "MAX_LOG_FILES": MAX_LOG_FILES,
        "USE_ROLLING_AVG": USE_ROLLING_AVG,
        "ROLLING_WINDOW": ROLLING_WINDOW,
        "PRINT_ONLY_ACTIONABLE": PRINT_ONLY_ACTIONABLE,
        "EV_THRESHOLD": EV_THRESHOLD,
        "MIN_KELLY_STAKE": MIN_KELLY_STAKE,
        "ML_MODEL_FILE_H5": str(ML_MODEL_FILE_H5),
        "OU_MODEL_FILE_H5": str(OU_MODEL_FILE_H5),
        "XGB_ML_MODEL_FILE": str(XGB_ML_MODEL_FILE),
        "XGB_OU_MODEL_FILE": str(XGB_OU_MODEL_FILE),
    }

def validate_config():
    """Check if critical files and directories exist, log warnings if missing."""
    issues = []
    if not BASE_DATA_DIR.exists():
        issues.append(f"Missing data directory: {BASE_DATA_DIR}")
    if not BASE_MODELS_DIR.exists():
        issues.append(f"Missing models directory: {BASE_MODELS_DIR}")
    if not (BASE_MODELS_DIR / "NN_Models").exists():
        issues.append(f"Missing NN_Models subfolder: {BASE_MODELS_DIR / 'NN_Models'}")
    if not (BASE_MODELS_DIR / "XGBoost").exists():
        issues.append(f"Missing XGBoost subfolder: {BASE_MODELS_DIR / 'XGBoost'}")
    if not ARCHIVE_DIR.exists():
        issues.append(f"Missing archive directory: {ARCHIVE_DIR}")
    if not LOG_FILE.parent.exists():
        issues.append(f"Missing logs directory: {LOG_FILE.parent}")
    if not ML_MODEL_FILE_H5.exists():
        issues.append(f"Missing ML model file: {ML_MODEL_FILE_H5}")
    if not OU_MODEL_FILE_H5.exists():
        issues.append(f"Missing OU model file: {OU_MODEL_FILE_H5}")
    if not XGB_ML_MODEL_FILE.exists():
        issues.append(f"Missing XGB ML model file: {XGB_ML_MODEL_FILE}")
    if not XGB_OU_MODEL_FILE.exists():
        issues.append(f"Missing XGB OU model file: {XGB_OU_MODEL_FILE}")

    if issues:
        for issue in issues:
            logging.warning(issue)
    else:
        logging.info("✅ Config validation passed — all critical files and directories exist.")

    mode = "rolling" if USE_ROLLING_AVG else "season"
    logging.info(f"📊 Feature building mode: {mode} averages (window={ROLLING_WINDOW if USE_ROLLING_AVG else 'N/A'})")

# Logging setup
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Model artifacts
MODEL_FILE_PKL = BASE_MODELS_DIR / "game_predictor.pkl"
MODEL_FILE_H5 = BASE_MODELS_DIR / "game_predictor.h5"

# Neural Network models
ML_MODEL_FILE_H5 = BASE_MODELS_DIR / "NN_Models" / "Trained-Model-ML.h5"
OU_MODEL_FILE_H5 = BASE_MODELS_DIR / "NN_Models" / "Trained-Model-OU.h5"

# XGBoost models
XGB_MODELS_DIR = BASE_MODELS_DIR / "XGBoost"
XGB_ML_MODEL_FILE = XGB_MODELS_DIR / "Trained-XGB-ML.json"
XGB_OU_MODEL_FILE = XGB_MODELS_DIR / "Trained-XGB-OU.json"

# Data files
TRAINING_FEATURES_FILE = BASE_DATA_DIR / "training_features.csv"
PLAYER_FEATURES_FILE = BASE_DATA_DIR / "player_features.csv"   # NEW
NEW_GAMES_FILE = BASE_DATA_DIR / "new_games.csv"
NEW_GAMES_FEATURES_FILE = BASE_DATA_DIR / "new_games_features.csv"
HISTORICAL_GAMES_FILE = BASE_DATA_DIR / "historical_games.csv"
PLAYER_STATS_FILE = BASE_DATA_DIR / "player_stats.csv"
GAME_RESULTS_FILE = BASE_DATA_DIR / "game_results.csv"

# Aliases
FEATURES_FILE = NEW_GAMES_FEATURES_FILE
FEATURES_DIR = BASE_DATA_DIR

# Results files
PREDICTIONS_FILE = BASE_RESULTS_DIR / "today_predictions.csv"
BANKROLL_FILE_TEMPLATE = BASE_RESULTS_DIR / "picks_bankroll_{model_type}.csv"
PICKS_FILE = BASE_RESULTS_DIR / "picks.csv"
PICKS_LOG = BASE_LOGS_DIR / "picks.log"
SUMMARY_FILE = BASE_RESULTS_DIR / "summary.csv"

# General settings
SEED = 42
DEFAULT_THRESHOLD = 0.6
DEFAULT_BANKROLL = 1000.0
MAX_KELLY_FRACTION = 0.05

# Actionable bet display filters
PRINT_ONLY_ACTIONABLE = os.getenv("PRINT_ONLY_ACTIONABLE", "false").lower() in ("1", "true", "yes")
EV_THRESHOLD = float(os.getenv("EV_THRESHOLD", "0.01"))
MIN_KELLY_STAKE = float(os.getenv("MIN_KELLY_STAKE", "1.0"))

# Cleanup / Rotation Settings
CLEANUP_MODE = os.getenv("CLEANUP_MODE", "archive").lower()
ARCHIVE_RETENTION_DAYS = int(os.getenv("ARCHIVE_RETENTION_DAYS", "180"))
MAX_DASHBOARD_IMAGES = int(os.getenv("MAX_DASHBOARD_IMAGES", "10"))
MAX_LOG_FILES = int(os.getenv("MAX_LOG_FILES", "20"))

# Feature building settings
USE_ROLLING_AVG = os.getenv("USE_ROLLING_AVG", "true").lower() in ("1", "true", "yes")
ROLLING_WINDOW = int(os.getenv("ROLLING_WINDOW", "5"))