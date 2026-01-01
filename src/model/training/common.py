from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Shared Training Logic
# File: src/model/training/common.py
# Author: Sadiq
#
# Description:
#     Shared training logic for all model types:
#       - model factory
#       - training
#       - optional calibration
#       - prediction on test set
#       - metric computation
# ============================================================

import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator
from loguru import logger

from src.model.training.hyperparams import (
    XGBOOST_PARAMS,
    LIGHTGBM_PARAMS,
    LOGISTIC_REGRESSION_PARAMS,
)
from src.model.training.metrics import compute_metrics


# ------------------------------------------------------------
# Model factory
# ------------------------------------------------------------

def create_model(model_type: str, family: str) -> BaseEstimator:
    """
    Create a model instance based on model_type and family.

    family: "xgboost" | "lightgbm" | "logistic_regression"
    """
    family = family.lower()

    if family == "xgboost":
        return (
            xgb.XGBClassifier(**XGBOOST_PARAMS)
            if model_type == "moneyline"
            else xgb.XGBRegressor(**XGBOOST_PARAMS)
        )

    if family == "lightgbm":
        return (
            lgb.LGBMClassifier(**LIGHTGBM_PARAMS)
            if model_type == "moneyline"
            else lgb.LGBMRegressor(**LIGHTGBM_PARAMS)
        )

    if family == "logistic_regression":
        if model_type != "moneyline":
            raise ValueError("Logistic regression only valid for moneyline models.")
        return LogisticRegression(**LOGISTIC_REGRESSION_PARAMS)

    raise ValueError(f"Unsupported model family: {family}")


# ------------------------------------------------------------
# Shared training logic
# ------------------------------------------------------------

def train_model_common(
    model_type: str,
    X_train,
    y_train,
    X_test,
    model_family: str = "xgboost",
    calibrate: bool = True,
):
    """
    Train a model and return:
        - trained model
        - predictions or probabilities on X_test
        - metrics dict
    """

    logger.info(f"Training {model_type} using {model_family}")

    model = create_model(model_type, model_family)
    logger.debug(f"Model hyperparameters: {model.get_params()}")

    model.fit(X_train, y_train)

    # Classification → optional calibration
    if model_type == "moneyline":
        if calibrate:
            logger.info("Applying probability calibration (CalibratedClassifierCV)")
            calibrated = CalibratedClassifierCV(model, cv=5)
            calibrated.fit(X_train, y_train)
            model = calibrated

        y_output = model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_true=y_train, y_pred=model.predict_proba(X_train)[:, 1])
        return model, y_output, metrics

    # Regression → predict values
    y_output = model.predict(X_test)
    metrics = compute_metrics(y_true=y_train, y_pred=model.predict(X_train))
    return model, y_output, metrics