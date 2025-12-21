# ============================================================
# Project: NBA Analytics & Betting Engine
# Author: Sadiq
# Description: Model hyperparameters and training configuration.
# ============================================================

MODEL_TYPE = "RandomForest"

RANDOM_FOREST_PARAMS = {
    "n_estimators": 400,
    "max_depth": 10,
    "min_samples_split": 4,
    "min_samples_leaf": 2,
    "random_state": 42,
}

TRAIN_TEST_SPLIT = 0.2
MIN_GAMES_FOR_RETRAIN = 200
