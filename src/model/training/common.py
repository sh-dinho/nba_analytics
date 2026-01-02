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
# ============================================================
from __future__ import annotations

import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from loguru import logger

from src.model.training.hyperparams import (
    XGBOOST_CLASSIFICATION_PARAMS,
    XGBOOST_REGRESSION_PARAMS,
    LIGHTGBM_CLASSIFICATION_PARAMS,
    LIGHTGBM_REGRESSION_PARAMS,
    LOGISTIC_REGRESSION_PARAMS,
)


def create_model(model_type: str, family: str):
    family = family.lower()
    is_classifier = model_type == "moneyline"

    if family == "xgboost":
        return (
            xgb.XGBClassifier(**XGBOOST_CLASSIFICATION_PARAMS)
            if is_classifier
            else xgb.XGBRegressor(**XGBOOST_REGRESSION_PARAMS)
        )

    if family == "lightgbm":
        return (
            lgb.LGBMClassifier(**LIGHTGBM_CLASSIFICATION_PARAMS)
            if is_classifier
            else lgb.LGBMRegressor(**LIGHTGBM_REGRESSION_PARAMS)
        )

    if family == "logistic_regression":
        if not is_classifier:
            raise ValueError("Logistic regression only valid for moneyline.")
        return LogisticRegression(**LOGISTIC_REGRESSION_PARAMS)

    raise ValueError(f"Unsupported model family: {family}")


def train_model_common(model_type, x_train, y_train, x_test, model_family="xgboost"):
    logger.info(f"Training {model_type} model using {model_family}...")

    model = create_model(model_type, model_family)

    logger.info("Fitting base model...")
    model.fit(x_train, y_train)

    if model_type == "moneyline":
        logger.info("Calibrating probability model...")
        calibrated = CalibratedClassifierCV(model, cv=5, method="sigmoid")
        calibrated.fit(x_train, y_train)
        model = calibrated
        y_pred = model.predict_proba(x_test)[:, 1]
    else:
        y_pred = model.predict(x_test)

    logger.info(f"Finished training {model_type} model.")
    return model, y_pred
