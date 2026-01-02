from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Totals Training Wrapper
# File: src/model/training/totals.py
# Author: Sadiq
#
# Description:
#     Thin wrapper around shared training logic for the
#     totals (predicted total points) regression model.
#     Now includes:
#       • full metrics report
#       • residual diagnostics
#       • model metadata
# ============================================================

from loguru import logger

from src.model.training.common import train_model_common
from src.model.training.full_metrics import full_metrics_report


def train_totals(
    X_train,
    y_train,
    X_test,
    y_test,
    model_family: str = "xgboost",
):
    """
    Train the totals (predicted total points) regression model.

    Returns:
        model   → trained model
        y_pred  → predicted totals
        report  → full regression metrics + residual diagnostics
    """
    logger.info(f"🏀 Training totals model using family='{model_family}'")
    logger.debug(
        f"Shapes: X_train={getattr(X_train, 'shape', None)}, "
        f"X_test={getattr(X_test, 'shape', None)}"
    )

    # --------------------------------------------------------
    # Train model (shared logic)
    # --------------------------------------------------------
    model, y_pred = train_model_common(
        model_type="totals",
        x_train=X_train,
        y_train=y_train,
        x_test=X_test,
        model_family=model_family,
    )

    # --------------------------------------------------------
    # Full regression metrics + residual diagnostics
    # --------------------------------------------------------
    report = full_metrics_report(
        model_type="totals",
        y_true=y_test,
        y_output=y_pred,
    )

    logger.success("Totals model training complete.")
    logger.info(
        f"Metrics summary: rmse={report['regression']['rmse']:.4f}, "
        f"mae={report['regression']['mae']:.4f}, "
        f"r2={report['regression']['r2']:.4f}"
    )

    return model, y_pred, report


