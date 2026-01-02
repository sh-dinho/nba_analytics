from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Hyperparameters (Tuned)
# File: src/model/training/hyperparams.py
# Author: Sadiq
#
# Description:
#     Tuned hyperparameters for NBA modeling.
#     Split into classification vs regression to avoid
#     accidental objective contamination.
# ============================================================

from typing import Dict, Any

HyperParams = Dict[str, Any]

# ------------------------------------------------------------
# XGBoost — Classification (Moneyline)
# ------------------------------------------------------------
XGBOOST_CLASSIFICATION_PARAMS: HyperParams = {
    "n_estimators": 600,
    "learning_rate": 0.025,
    "max_depth": 4,
    "min_child_weight": 3,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "gamma": 0.1,
    "reg_lambda": 1.2,
    "reg_alpha": 0.1,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "n_jobs": -1,
    "verbosity": 0,
}

# ------------------------------------------------------------
# XGBoost — Regression (Totals, Spread)
# ------------------------------------------------------------
XGBOOST_REGRESSION_PARAMS: HyperParams = {
    "n_estimators": 600,
    "learning_rate": 0.025,
    "max_depth": 4,
    "min_child_weight": 3,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "gamma": 0.1,
    "reg_lambda": 1.2,
    "reg_alpha": 0.1,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "n_jobs": -1,
    "verbosity": 0,
}

# ------------------------------------------------------------
# LightGBM — Classification
# ------------------------------------------------------------
LIGHTGBM_CLASSIFICATION_PARAMS: HyperParams = {
    "n_estimators": 700,
    "learning_rate": 0.02,
    "num_leaves": 48,
    "max_depth": -1,
    "min_child_samples": 25,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_lambda": 1.0,
    "reg_alpha": 0.1,
    "n_jobs": -1,
    "verbose": -1,
}

# ------------------------------------------------------------
# LightGBM — Regression
# ------------------------------------------------------------
LIGHTGBM_REGRESSION_PARAMS: HyperParams = {
    "n_estimators": 700,
    "learning_rate": 0.02,
    "num_leaves": 48,
    "max_depth": -1,
    "min_child_samples": 25,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_lambda": 1.0,
    "reg_alpha": 0.1,
    "n_jobs": -1,
    "verbose": -1,
}

# ------------------------------------------------------------
# Logistic Regression (Moneyline only)
# ------------------------------------------------------------
LOGISTIC_REGRESSION_PARAMS: HyperParams = {
    "max_iter": 500,
    "solver": "lbfgs",
    "C": 0.7,
    "n_jobs": -1,
}

# ------------------------------------------------------------
# Unified map (optional)
# ------------------------------------------------------------
MODEL_FAMILY_DEFAULTS = {
    "xgboost_classification": XGBOOST_CLASSIFICATION_PARAMS,
    "xgboost_regression": XGBOOST_REGRESSION_PARAMS,
    "lightgbm_classification": LIGHTGBM_CLASSIFICATION_PARAMS,
    "lightgbm_regression": LIGHTGBM_REGRESSION_PARAMS,
    "logistic_regression": LOGISTIC_REGRESSION_PARAMS,
}

__all__ = [
    "XGBOOST_CLASSIFICATION_PARAMS",
    "XGBOOST_REGRESSION_PARAMS",
    "LIGHTGBM_CLASSIFICATION_PARAMS",
    "LIGHTGBM_REGRESSION_PARAMS",
    "LOGISTIC_REGRESSION_PARAMS",
    "MODEL_FAMILY_DEFAULTS",
]
