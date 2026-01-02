from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Moneyline Training Wrapper
# File: src/model/training/moneyline.py
# Author: Sadiq
#
# Description:
#     Thin wrapper around shared training logic for the
#     moneyline (win probability) model.
#     Now includes:
#       • full metrics report
#       • model metadata
#       • safer logging
# ============================================================

from loguru import logger

from src.model.training.common import train_model_common
from src.model.training.full_metrics import full_metrics_report


def train_moneyline(
    X_train,
    y_train,
    X_test,
    y_test,
    model_family: str = "xgboost",
):
    """
    Train the moneyline (win probability) model.

    Returns:
        model        → trained + calibrated model
        y_pred       → predicted probabilities
        report       → full metrics report (classification + calibration + thresholds)
    """
    logger.info(f"🏀 Training moneyline model using family='{model_family}'")
    logger.debug(
        f"Shapes: X_train={getattr(X_train, 'shape', None)}, "
        f"X_test={getattr(X_test, 'shape', None)}"
    )

    # --------------------------------------------------------
    # Train model (shared logic handles calibration)
    # --------------------------------------------------------
    model, y_pred = train_model_common(
        model_type="moneyline",
        x_train=X_train,
        y_train=y_train,
        x_test=X_test,
        model_family=model_family,
    )

    # --------------------------------------------------------
    # Full metrics report (classification + calibration + thresholds)
    # --------------------------------------------------------
    report = full_metrics_report(
        model_type="moneyline",
        y_true=y_test,
        y_output=y_pred,
    )

    logger.success("Moneyline model training complete.")
    logger.info(f"Metrics summary: accuracy={report['classification']['accuracy']:.4f}, "
                f"brier={report['classification']['brier']:.4f}, "
                f"auc={report['classification']['auc']:.4f}")

    return model, y_pred, report
