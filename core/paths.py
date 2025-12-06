# ============================================================
# File: core/paths.py
# Purpose: Centralized path definitions for NBA analytics project
#          - Manages paths for input, output, and model directories
#          - Ensures directory structure consistency
# ============================================================

from pathlib import Path
import datetime

# ---------------- Base Directories ----------------
BASE_DIR = Path(__file__).resolve().parent.parent        # Base project directory
DATA_DIR = BASE_DIR / "Data"                             # Directory for raw data and features
RESULTS_DIR = BASE_DIR / "Results"                       # Directory for storing results, logs, and summaries
LOGS_DIR = RESULTS_DIR / "logs"                          # Subdirectory for log files
ARCHIVE_DIR = DATA_DIR / "archive"                       # Directory for archived data files
MODELS_DIR = BASE_DIR / "Models"                         # Directory for model files (e.g., trained models)

# Ensure the base directories exist
for d in [DATA_DIR, RESULTS_DIR, LOGS_DIR, ARCHIVE_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------- Feature Files ----------------
TRAINING_FEATURES_FILE = DATA_DIR / "training_features.csv"
NEW_GAMES_FEATURES_FILE = DATA_DIR / "new_games_features.csv"
PLAYER_FEATURES_FILE = DATA_DIR / "player_features.csv"

# Dynamically select the feature file that exists
if TRAINING_FEATURES_FILE.exists():
    FEATURES_FILE = TRAINING_FEATURES_FILE
elif NEW_GAMES_FEATURES_FILE.exists():
    FEATURES_FILE = NEW_GAMES_FEATURES_FILE
else:
    FEATURES_FILE = None  # If neither file exists, set FEATURES_FILE to None

# ---------------- Picks Workflow ----------------
TODAY_PREDICTIONS_FILE = RESULTS_DIR / "today_predictions.csv"    # Today's predictions
PICKS_FILE = RESULTS_DIR / "picks.csv"                             # File to store picks
PICKS_LOG = LOGS_DIR / "picks_log.csv"                             # Log of picks made
PICKS_SUMMARY_FILE = RESULTS_DIR / "picks_summary.csv"             # Summary of picks made
PICKS_BANKROLL_FILE = RESULTS_DIR / "picks_bankroll.csv"           # Bankroll tracking for picks

# Date-stamped versions of picks summary and bankroll files for the current day
today_str = datetime.date.today().strftime("%Y-%m-%d")
DAILY_PICKS_SUMMARY_FILE = RESULTS_DIR / f"picks_summary_{today_str}.csv"
DAILY_PICKS_BANKROLL_FILE = RESULTS_DIR / f"picks_bankroll_{today_str}.csv"

# ---------------- Summary Files ----------------
SUMMARY_FILE = RESULTS_DIR / "summary.csv"
MONTHLY_SUMMARY_FILE = RESULTS_DIR / "monthly_summary.csv"
PIPELINE_SUMMARY_FILE = RESULTS_DIR / "pipeline_summary.csv"
DOWNLOAD_SUMMARY_FILE = RESULTS_DIR / "download_summary.csv"

# ---------------- Model Files ----------------
TEAM_MODEL_FILE = MODELS_DIR / "team_model.pkl"
PLAYER_MODEL_FILE = MODELS_DIR / "player_model.pkl"
XGB_ML_MODEL_FILE = MODELS_DIR / "xgb_ml_model.pkl"
XGB_OU_MODEL_FILE = MODELS_DIR / "xgb_ou_model.pkl"
ENSEMBLE_MODEL_FILE = MODELS_DIR / "ensemble_model.pkl"

# ---------------- Encoders ----------------
TEAM_ENCODING_FILE = MODELS_DIR / "team_encoding_map.json"

# ---------------- Raw Data Files ----------------
HISTORICAL_GAMES_FILE = DATA_DIR / "historical_games.csv"
NEW_GAMES_FILE = DATA_DIR / "new_games.csv"
PLAYER_GAMES_FILE = DATA_DIR / "player_games.csv"

# ---------------- AI Tracker ----------------
AI_TRACKER_DIR = RESULTS_DIR / "ai_tracker"
AI_TRACKER_TEAMS_FILE = AI_TRACKER_DIR / "teams.csv"                   # Team-level AI tracking
AI_TRACKER_PLAYERS_FILE = AI_TRACKER_DIR / "players.csv"               # Player-level AI tracking
AI_TRACKER_INSIGHT_FILE = AI_TRACKER_DIR / "insight.txt"               # Insight report for teams/players
AI_TRACKER_DASHBOARD_FILE = AI_TRACKER_DIR / "team_dashboard.png"      # AI-generated team dashboard image
AI_TRACKER_PLAYER_DASHBOARD_FILE = AI_TRACKER_DIR / "player_dashboard.png"  # AI-generated player dashboard image
AI_TRACKER_SUMMARY_FILE = AI_TRACKER_DIR / "summary.csv"               # Summary file of AI tracking

# ---------------- Utility (updated) ----------------
def ensure_dirs(strict: bool = False):
    """Ensure all required directories exist."""
    for d in [DATA_DIR, RESULTS_DIR, LOGS_DIR, ARCHIVE_DIR, MODELS_DIR, AI_TRACKER_DIR]:
        if not d.exists():
            if strict:
                raise FileNotFoundError(f"Required directory missing: {d}")
            d.mkdir(parents=True, exist_ok=True)
