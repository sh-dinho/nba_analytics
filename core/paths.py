# ============================================================
# File: core/paths.py
# Purpose: Centralized file and directory paths for NBA analytics pipeline
# ============================================================

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Directories
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
ARCHIVE_DIR = PROJECT_ROOT / "archive"

# Ensure directories exist
for d in [DATA_DIR, LOGS_DIR, RESULTS_DIR, MODELS_DIR, ARCHIVE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# === Model files ===
# Centralized references for all ML models used in the pipeline

# Legacy pickle model (if still needed for backward compatibility)
MODEL_FILE_PKL = MODELS_DIR / "xgb_model.pkl"

# XGBoost models (JSON format)
XGB_ML_MODEL_FILE = MODELS_DIR / "xgb_ml.json"   # Moneyline model
XGB_OU_MODEL_FILE = MODELS_DIR / "xgb_ou.json"   # Over/Under model

# Optional: add ensemble or other specialized models here
ENSEMBLE_MODEL_FILE = MODELS_DIR / "ensemble_model.json"
PLAYER_MODEL_FILE = MODELS_DIR / "player_model.json"
TEAM_MODEL_FILE = MODELS_DIR / "team_model.json"

# === Data files ===
HISTORICAL_GAMES_FILE = DATA_DIR / "historical_games.csv"
PLAYER_STATS_FILE = DATA_DIR / "player_stats.csv"
NEW_GAMES_FILE = DATA_DIR / "new_games.csv"

# === Logs ===
FEATURES_LOG_FILE = LOGS_DIR / "features_summary.csv"
CONFIG_LOG_FILE = LOGS_DIR / "config_summary.csv"

# === Results ===
SUMMARY_FILE = RESULTS_DIR / "pipeline_summary.csv"
PICKS_FILE = RESULTS_DIR / "picks.csv"
PICKS_BANKROLL_FILE = RESULTS_DIR / "bankroll.csv"