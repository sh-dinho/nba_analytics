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
# Metrics to compute and log during training
EVAL_METRICS = ["accuracy", "f1", "roc_auc"]

# Thresholds for classification confidence (optional)
CONFIDENCE_THRESHOLDS = {
    "high": 0.75,
    "medium": 0.55,
    "low": 0.50,
}

# ---------------- Bankroll ----------------
# Default bankroll for the picks (used in stake calculations)
DEFAULT_BANKROLL = 1000.0  # You can change this value to whatever makes sense for your project

# ---------------- Kelly Fraction ----------------
# Maximum Kelly Fraction to determine stake size for bets
MAX_KELLY_FRACTION = 0.05  # You can adjust this as needed

# ---------------- EV Threshold ----------------
# Minimum expected value (EV) for a pick to be considered worth betting
EV_THRESHOLD = 0.05  # You can adjust this threshold as needed

# ---------------- Minimum Kelly Stake ----------------
# Minimum stake allowed for any bet, to avoid placing unreasonably small bets
MIN_KELLY_STAKE = 10.0  # You can adjust this minimum stake as needed

# ---------------- Paths ----------------
HISTORICAL_GAMES_FILE = Path("path/to/your/historical_games.csv")  # Adjust this path to your actual CSV file
PLAYER_GAMES_FILE = Path("path/to/your/player_games.csv")  # Same for player games file

# Feature Files
TRAINING_FEATURES_FILE = Path("path/to/your/training_features.csv")  # Path for saving team features
PLAYER_FEATURES_FILE = Path("path/to/your/player_features.csv")  # Path for saving player features

# ---------------- Logging ----------------
def log_config_snapshot():
    """Log current configuration values for reproducibility."""
    from core.log_config import init_global_logger
    logger = init_global_logger()
    logger.info(
        f"Config snapshot → "
        f"USE_ROLLING_AVG={USE_ROLLING_AVG}, "
        f"ROLLING_WINDOW={ROLLING_WINDOW}, "
        f"RANDOM_SEED={RANDOM_SEED}, "
        f"TEST_SIZE={TEST_SIZE}, "
        f"EVAL_METRICS={EVAL_METRICS}"
    )
