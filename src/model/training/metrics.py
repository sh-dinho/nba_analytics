from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Training Metrics
# File: src/model/training/metrics.py
# Author: Sadiq
#
# Description:
#     Centralized metrics for all model types.
#     - Classification (moneyline)
#     - Regression (totals, spread)
#     - Safe, dashboard‑ready, monitoring‑ready
# ============================================================

import numpy as np
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
)


# ------------------------------------------------------------
# Classification metrics (moneyline)
# ------------------------------------------------------------

def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Compute classification metrics for moneyline models.

    y_prob: predicted probabilities for class 1
    threshold: decision threshold for accuracy
    """
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {}

    # Accuracy
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))

    # Precision / Recall / F1 (safe)
    try:
        metrics["precision"] = float(precision_score(y_true, y_pred))
        metrics["recall"] = float(recall_score(y_true, y_pred))
        metrics["f1"] = float(f1_score(y_true, y_pred))
    except Exception:
        logger.warning("Precision/Recall/F1 undefined; setting to NaN")
        metrics["precision"] = metrics["recall"] = metrics["f1"] = float("nan")

    # Brier score
    metrics["brier"] = float(brier_score_loss(y_true, y_prob))

    # Log loss (safe)
    try:
        metrics["log_loss"] = float(log_loss(y_true, y_prob))
    except ValueError:
        logger.warning("log_loss undefined for constant y_true; setting to NaN")
        metrics["log_loss"] = float("nan")

    # AUC (safe)
    try:
        metrics["auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        logger.warning("AUC undefined for constant y_true; setting to NaN")
        metrics["auc"] = float("nan")

    return metrics


# ------------------------------------------------------------
# Regression metrics (totals, spread)
# ------------------------------------------------------------

def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """
    Compute regression metrics for totals/spread models.
    """
    metrics = {
        "rmse": float(mean_squared_error(y_true, y_pred, squared=False)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "explained_variance": float(explained_variance_score(y_true, y_pred)),
    }

    # MAPE (safe)
    try:
        mape = np.mean(np.abs((y_true - y_pred) / y_true))
        metrics["mape"] = float(mape)
    except Exception:
        logger.warning("MAPE undefined (division by zero); setting to NaN")
        metrics["mape"] = float("nan")

    return metrics


# ------------------------------------------------------------
# Unified interface
# ------------------------------------------------------------

def compute_metrics(
    model_type: str,
    y_true: np.ndarray,
    y_output: np.ndarray,
) -> dict:
    """
    Unified metric computation for all model types.

    model_type:
        - "moneyline" → classification
        - "totals" → regression
        - "spread" → regression

    y_output:
        - probabilities for moneyline
        - predictions for totals/spread
    """
    if model_type == "moneyline":
        return compute_classification_metrics(y_true, y_output)

    if model_type in ("totals", "spread"):
        return compute_regression_metrics(y_true, y_output)

    raise ValueError(f"Unsupported model_type: {model_type}")
