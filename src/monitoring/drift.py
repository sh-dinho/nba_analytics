from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Drift Detection Utilities
# File: src/monitoring/drift.py
# Author: Sadiq
#
# Description:
#     Statistical drift detection utilities for monitoring
#     model inputs and feature distributions.
#
#     Includes:
#       • KS-test drift detection
#       • PSI (Population Stability Index)
#       • Drift summarization utilities
#
#     Used in:
#       • Daily feature monitoring
#       • Model retraining triggers
#       • Data quality dashboards
# ============================================================

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from loguru import logger

from src.config.monitoring import MONITORING

DEFAULT_ALPHA = MONITORING["drift_alpha"]
MIN_SAMPLES = MONITORING["min_samples"]
PSI_BUCKETS = MONITORING.get("psi_buckets", 10)


# ------------------------------------------------------------
# KS-Test Drift Detection
# ------------------------------------------------------------

def ks_drift_report(
    baseline: pd.DataFrame,
    recent: pd.DataFrame,
    columns: List[str],
    alpha: float = DEFAULT_ALPHA,
    min_samples: int = MIN_SAMPLES,
) -> Dict[str, Dict[str, float]]:
    """
    Perform KS-test drift detection for each column.

    Returns:
        {
            column_name: {
                "statistic": float,
                "pvalue": float,
                "drift": 0.0 or 1.0
            }
        }
    """
    report: Dict[str, Dict[str, float]] = {}

    for col in columns:
        if col not in baseline.columns or col not in recent.columns:
            logger.warning(f"[KS] Column missing in baseline or recent: {col}")
            report[col] = {"statistic": np.nan, "pvalue": np.nan, "drift": np.nan}
            continue

        b = baseline[col].dropna().values
        r = recent[col].dropna().values

        if len(b) < min_samples or len(r) < min_samples:
            logger.warning(
                f"[KS] Insufficient samples for {col} "
                f"(baseline={len(b)}, recent={len(r)})"
            )
            report[col] = {"statistic": np.nan, "pvalue": np.nan, "drift": np.nan}
            continue

        stat, pval = ks_2samp(b, r)
        drift_flag = float(pval < alpha)

        report[col] = {
            "statistic": float(stat),
            "pvalue": float(pval),
            "drift": drift_flag,
        }

    return report


# ------------------------------------------------------------
# PSI (Population Stability Index)
# ------------------------------------------------------------

def _psi_single(baseline: np.ndarray, recent: np.ndarray, buckets: int) -> float:
    """
    Compute PSI for a single feature.
    """
    try:
        quantiles = np.linspace(0, 1, buckets + 1)
        cuts = np.quantile(baseline, quantiles)

        b_counts, _ = np.histogram(baseline, bins=cuts)
        r_counts, _ = np.histogram(recent, bins=cuts)

        b_perc = b_counts / len(baseline)
        r_perc = r_counts / len(recent)

        # Avoid division by zero
        b_perc = np.where(b_perc == 0, 1e-6, b_perc)
        r_perc = np.where(r_perc == 0, 1e-6, r_perc)

        psi = np.sum((b_perc - r_perc) * np.log(b_perc / r_perc))
        return float(psi)

    except Exception as e:
        logger.error(f"[PSI] Failed to compute PSI: {e}")
        return np.nan


def psi_report(
    baseline: pd.DataFrame,
    recent: pd.DataFrame,
    columns: List[str],
    buckets: int = PSI_BUCKETS,
    min_samples: int = MIN_SAMPLES,
) -> Dict[str, float]:
    """
    Compute PSI for each column.

    Returns:
        { column_name: psi_value }
    """
    report: Dict[str, float] = {}

    for col in columns:
        if col not in baseline.columns or col not in recent.columns:
            logger.warning(f"[PSI] Column missing in baseline or recent: {col}")
            report[col] = np.nan
            continue

        b = baseline[col].dropna().values
        r = recent[col].dropna().values

        if len(b) < min_samples or len(r) < min_samples:
            logger.warning(
                f"[PSI] Insufficient samples for {col} "
                f"(baseline={len(b)}, recent={len(r)})"
            )
            report[col] = np.nan
            continue

        report[col] = _psi_single(b, r, buckets)

    return report


# ------------------------------------------------------------
# Drift Summary
# ------------------------------------------------------------

def summarize_drift(report: Dict[str, Dict[str, float]]) -> Tuple[int, int]:
    """
    Summarize KS-test drift report.

    Returns:
        (num_drifted_columns, num_tested_columns)
    """
    tested = [c for c, v in report.items() if not np.isnan(v.get("pvalue", np.nan))]
    drifted = [c for c in tested if report[c]["drift"] == 1.0]
    return len(drifted), len(tested)