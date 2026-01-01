from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Moneyline Training Wrapper
# File: src/model/training/moneyline.py
# Author: Sadiq
#
# Description:
#     Thin wrapper around the shared training logic for
#     training the moneyline (win probability) model.
# ============================================================

from loguru import logger
from typing import Tuple, Any

from src.model.training.common import train_model_common


def train_moneyline(
    X_train,
    y_train,
    X_test,
    model_family: str = "xgboost",
    calibrate: bool = True,
):
    """
    Train the moneyline (win probability) model.

    Parameters
    ----------
    X_train : array-like
        Training feature matrix.
    y_train : array-like
        Training labels (0/1).
    X_test : array-like
        Test feature matrix.
    model_family : str
        "xgboost" | "lightgbm" | "logistic_regression"
    calibrate : bool
        Whether to apply probability calibration.

    Returns
    -------
    model : trained model object
    y_output : predictions or probabilities on X_test
    metrics : dict
        Training metrics computed on the training set.
    """
    logger.info(
        f"Training moneyline model using {model_family} "
        f"(calibrate={calibrate})"
    )
    logger.debug(f"X_train={getattr(X_train, 'shape', None)}, "
                 f"X_test={getattr(X_test, 'shape', None)}")

    return train_model_common(
        model_type="moneyline",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        model_family=model_family,
        calibrate=calibrate,
    )