# ============================================================
# File: core/config.py
# Purpose: Global configuration for NBA analytics project
# ============================================================

from pathlib import Path

# ---------------- Feature Engineering ----------------
USE_ROLLING_AVG = True          # If True, use rolling averages; if False, use season averages
ROLLING_WINDOW = 5              # Number of games to use for rolling averages

# ---------------- Model Training ----------------
RANDOM_SEED = 42                # Default random seed
TEST_SIZE = 0.2                 # Default test split ratio

DEFAULT_XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_SEED,
    "use_label_encoder": False,       # Recommended for modern XGBoost
    "eval_metric": "logloss",         # Avoids warning about default metric
    "enable_categorical": True,       # Enables categorical handling
}

DEFAULT_RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": None,
    "random_state": RANDOM_SEED,
}

DEFAULT_LOGREG_PARAMS = {
    "max_iter": 1000,
    "solver": "lbfgs",
    "random_state": RANDOM_SEED,
}

# ---------------- Evaluation Settings ----------------
EVAL_METRICS = ["accuracy", "f1", "roc_auc"]

CONFIDENCE_THRESHOLDS = {
    "high": 0.75,
    "medium": 0.55,
    "low": 0.50,
}

# ---------------- Bankroll ----------------
DEFAULT_BANKROLL = 1000.0

# ---------------- Kelly Fraction ----------------
MAX_KELLY_FRACTION = 0.05

# ---------------- EV Threshold ----------------
EV_THRESHOLD = 0.05

# ---------------- Minimum Kelly Stake ----------------
MIN_KELLY_STAKE = 10.0

# ---------------- Paths ----------------
ROOT_DIR = Path(__file__).parent.parent.resolve()
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
MODELS_DIR = ROOT_DIR / "models"

# Data files
HISTORICAL_GAMES_FILE = DATA_DIR / "historical_games.csv"
PLAYER_GAMES_FILE = DATA_DIR / "player_games.csv"
TRAINING_FEATURES_FILE = DATA_DIR / "training_features.csv"
PLAYER_FEATURES_FILE = DATA_DIR / "player_features.csv"

# Model files
XGB_ML_MODEL_FILE = MODELS_DIR / "xgb_ml.json"
XGB_OU_MODEL_FILE = MODELS_DIR / "xgb_ou.json"
ENSEMBLE_MODEL_FILE = MODELS_DIR / "ensemble_model.pkl"
TEAM_MODEL_FILE = MODELS_DIR / "team_model.pkl"
PLAYER_MODEL_FILE = MODELS_DIR / "player_model.pkl"

# Summary files
MONTHLY_SUMMARY_FILE = RESULTS_DIR / "monthly_summary.csv"
PIPELINE_SUMMARY_FILE = RESULTS_DIR / "pipeline_summary.csv"


USE_ROLLING_AVG = True
ROLLING_WINDOW = 5
RANDOM_SEED = 42
TEST_SIZE = 0.2
EVAL_METRICS = ["accuracy", "f1", "roc_auc"]

DEFAULT_BANKROLL = 1000

# Telegram settings
SEND_NOTIFICATIONS = True   # default toggle
TELEGRAM_BOT_TOKEN = "your-bot-token-here"
TELEGRAM_CHAT_ID = "your-chat-id-here"

# ---------------- Logging ----------------
def log_config_snapshot():
    """Log current configuration values for reproducibility."""
    from nba_core.log_config import init_global_logger
    logger = init_global_logger()
    logger.info(
        f"Config snapshot → "
        f"USE_ROLLING_AVG={USE_ROLLING_AVG}, "
        f"ROLLING_WINDOW={ROLLING_WINDOW}, "
        f"RANDOM_SEED={RANDOM_SEED}, "
        f"TEST_SIZE={TEST_SIZE}, "
        f"EVAL_METRICS={EVAL_METRICS}"
    )
