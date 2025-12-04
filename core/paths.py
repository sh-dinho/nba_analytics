# ============================================================
# File: core/paths.py
# Purpose: Centralized file and directory paths for NBA analytics pipeline
# ============================================================

from pathlib import Path
from datetime import date

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
MODEL_FILE_PKL = MODELS_DIR / "xgb_model.pkl"        # legacy pickle model
XGB_ML_MODEL_FILE = MODELS_DIR / "xgb_ml.json"       # moneyline model
XGB_OU_MODEL_FILE = MODELS_DIR / "xgb_ou.json"       # over/under model
ENSEMBLE_MODEL_FILE = MODELS_DIR / "ensemble.json"   # optional ensemble
PLAYER_MODEL_FILE = MODELS_DIR / "player.json"       # optional player-level
TEAM_MODEL_FILE = MODELS_DIR / "team.json"           # optional team-level

# === Data files ===
HISTORICAL_GAMES_FILE = DATA_DIR / "historical_games.csv"
PLAYER_STATS_FILE = DATA_DIR / "player_stats.csv"
NEW_GAMES_FILE = DATA_DIR / "new_games.csv"
FEATURES_FILE = DATA_DIR / "features.csv"            # ✅ added to fix import
# Logs
FEATURES_LOG_FILE = LOGS_DIR / "features_summary.csv"
CONFIG_LOG_FILE = LOGS_DIR / "config_summary.csv"

# === Results ===
SUMMARY_FILE = RESULTS_DIR / "pipeline_summary.csv"
PICKS_FILE = RESULTS_DIR / "picks.csv"
PICKS_BANKROLL_FILE = RESULTS_DIR / "bankroll.csv"

# Daily dashboard outputs
DAILY_DASHBOARD_CSV_TEMPLATE = RESULTS_DIR / f"daily_dashboard_{date.today().isoformat()}.csv"
DAILY_DASHBOARD_PNG_TEMPLATE = RESULTS_DIR / f"daily_dashboard_{date.today().isoformat()}.png"
