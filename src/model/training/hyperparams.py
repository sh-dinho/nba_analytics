from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Hyperparameters
# File: src/model/training/hyperparams.py
# Author: Sadiq
#
# Description:
#     Centralized hyperparameters for all model families.
#     Used by the model factory in training/common.py.
# ============================================================

from typing import Dict, Any

# ------------------------------------------------------------
# XGBoost
# ------------------------------------------------------------
XGBOOST_PARAMS: Dict[str, Any] = {
    # Balanced depth + learning rate for stable calibration
    "n_estimators": 500,
    "learning_rate": 0.03,
    "max_depth": 5,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "eval_metric": "logloss",  # works for both binary and regression
}

# ------------------------------------------------------------
# LightGBM
# ------------------------------------------------------------
LIGHTGBM_PARAMS: Dict[str, Any] = {
    # max_depth = -1 → unlimited depth (LightGBM default)
    "n_estimators": 500,
    "learning_rate": 0.03,
    "max_depth": -1,
    "num_leaves": 64,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
}

# ------------------------------------------------------------
# Logistic Regression
# ------------------------------------------------------------
LOGISTIC_REGRESSION_PARAMS: Dict[str, Any] = {
    "max_iter": 200,
    "solver": "lbfgs",
    "C": 1.0,
}

__all__ = [
    "XGBOOST_PARAMS",
    "LIGHTGBM_PARAMS",
    "LOGISTIC_REGRESSION_PARAMS",
]