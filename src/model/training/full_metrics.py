from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Full Metrics Report
# File: src/model/training/full_metrics.py
# Author: Sadiq
#
# Description:
#     Unified, dashboard-ready metrics report for all model types.
#     Includes:
#       • classification metrics
#       • regression metrics
#       • calibration diagnostics
#       • residual diagnostics
#       • threshold diagnostics
#       • metadata
# ============================================================

import numpy as np
from datetime import datetime
from loguru import logger

from src.model.training.metrics import (
    compute_classification_metrics,
    compute_regression_metrics,
)


# ------------------------------------------------------------
# Calibration diagnostics (ECE, MCE)
# ------------------------------------------------------------

def compute_calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10):
    """
    Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
    """
    try:
        bin_edges = np.linspace(0, 1, bins + 1)
        ece = 0.0
        mce = 0.0

        for i in range(bins):
            mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
            if mask.sum() == 0:
                continue

            avg_pred = y_prob[mask].mean()
            avg_true = y_true[mask].mean()
            error = abs(avg_pred - avg_true)

            ece += (mask.sum() / len(y_true)) * error
            mce = max(mce, error)

        return {"ece": float(ece), "mce": float(mce)}

    except Exception as e:
        logger.warning(f"Calibration metrics failed: {e}")
        return {"ece": float("nan"), "mce": float("nan")}


# ------------------------------------------------------------
# Residual diagnostics (regression only)
# ------------------------------------------------------------

def compute_residual_diagnostics(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Residual distribution statistics for regression models.
    """
    residuals = y_true - y_pred

    return {
        "residual_mean": float(np.mean(residuals)),
        "residual_std": float(np.std(residuals)),
        "residual_skew": float(np.mean((residuals - residuals.mean())**3) / (residuals.std()**3 + 1e-9)),
        "residual_kurtosis": float(np.mean((residuals - residuals.mean())**4) / (residuals.std()**4 + 1e-9)),
    }


# ------------------------------------------------------------
# Threshold diagnostics (classification only)
# ------------------------------------------------------------

def compute_threshold_diagnostics(y_true: np.ndarray, y_prob: np.ndarray):
    """
    Compute accuracy across a range of thresholds.
    Useful for dashboards and decision optimization.
    """
    thresholds = np.linspace(0.05, 0.95, 19)
    results = {}

    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        acc = (pred == y_true).mean()
        results[f"thr_{t:.2f}"] = float(acc)

    return results


# ------------------------------------------------------------
# Full metrics report
# ------------------------------------------------------------

def full_metrics_report(
    model_type: str,
    y_true: np.ndarray,
    y_output: np.ndarray,
) -> dict:
    """
    Unified, dashboard-ready metrics report for all model types.
    """
    report = {
        "model_type": model_type,
        "timestamp": datetime.utcnow().isoformat(),
        "n_samples": int(len(y_true)),
    }

    # --------------------------------------------------------
    # Classification (moneyline)
    # --------------------------------------------------------
    if model_type == "moneyline":
        report["classification"] = compute_classification_metrics(y_true, y_output)
        report["calibration"] = compute_calibration_metrics(y_true, y_output)
        report["thresholds"] = compute_threshold_diagnostics(y_true, y_output)
        return report

    # --------------------------------------------------------
    # Regression (spread, totals)
    # --------------------------------------------------------
    if model_type in ("spread", "totals"):
        report["regression"] = compute_regression_metrics(y_true, y_output)
        report["residuals"] = compute_residual_diagnostics(y_true, y_output)
        return report

    raise ValueError(f"Unsupported model_type: {model_type}")
