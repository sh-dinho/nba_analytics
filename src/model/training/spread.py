from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Spread Training Wrapper
# File: src/model/training/spread.py
# Author: Sadiq
#
# Description:
#     Thin wrapper around the shared training logic for
#     training the spread (scoring margin) regression model.
# ============================================================

from loguru import logger
from typing import Any

from src.model.training.common import train_model_common


def train_spread(
    X_train,
    y_train,
    X_test,
    model_family: str = "xgboost",
):
    """
    Train the spread (scoring margin) regression model.

    Parameters
    ----------
    X_train : array-like
        Training feature matrix.
    y_train : array-like
        Training regression targets (scoring margin).
    X_test : array-like
        Test feature matrix.
    model_family : str
        "xgboost" | "lightgbm"

    Returns
    -------
    model : trained model object
    y_output : predictions on X_test
    metrics : dict
        Training metrics computed on the training set.
    """
    logger.info(f"Training spread model using {model_family}")
    logger.debug(f"X_train={getattr(X_train, 'shape', None)}, "
                 f"X_test={getattr(X_test, 'shape', None)}")

    return train_model_common(
        model_type="spread",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        model_family=model_family,
        calibrate=False,  # regression models do not use calibration
    )