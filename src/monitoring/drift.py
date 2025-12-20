# ============================================================
# File: src/monitoring/drift.py
# Purpose: Data drift detection utilities (KS-test, PSI)
# Version: 3.0 (config-driven, extensible, robust)
# Author: Your Team
# Date: December 2025
# ============================================================

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ------------------------
# Minimal config placeholder
# ------------------------
class Config:
    monitoring = {
        "drift_alpha": 0.05,
        "min_samples": 20,
    }


cfg = Config()
DEFAULT_ALPHA = cfg.monitoring.get("drift_alpha", 0.05)
MIN_SAMPLES = cfg.monitoring.get("min_samples", 20)


# ------------------------------------------------------------
# KS Drift Detection
# ------------------------------------------------------------
def ks_drift_report(
    baseline: pd.DataFrame,
    recent: pd.DataFrame,
    columns: List[str],
    alpha: float = DEFAULT_ALPHA,
    min_samples: int = MIN_SAMPLES,
) -> Dict[str, Dict[str, float]]:
    """
    Compute KS-test p-values for specified columns comparing baseline vs recent.

    Returns:
        {
            column: {
                "statistic": float,
                "pvalue": float,
                "drift": 1.0 or 0.0
            }
        }
    """
    report: Dict[str, Dict[str, float]] = {}

    for col in columns:
        if col not in baseline.columns or col not in recent.columns:
            logger.warning("Column %s missing in baseline or recent data", col)
            report[col] = {"statistic": np.nan, "pvalue": np.nan, "drift": np.nan}
            continue

        b = baseline[col].dropna().values
        r = recent[col].dropna().values

        if len(b) < min_samples or len(r) < min_samples:
            logger.warning(
                "Insufficient samples for KS test on %s (baseline=%d, recent=%d)",
                col,
                len(b),
                len(r),
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
# Drift Summary
# ------------------------------------------------------------
def summarize_drift(report: Dict[str, Dict[str, float]]) -> Tuple[int, int]:
    """
    Summarize drift report.

    Returns:
        (drift_columns_count, total_columns_tested)
    """
    tested = [
        col for col, vals in report.items() if not np.isnan(vals.get("pvalue", np.nan))
    ]
    drifted = [col for col in tested if report[col]["drift"] == 1.0]

    return len(drifted), len(tested)


# ------------------------------------------------------------
# PSI (Population Stability Index)
# ------------------------------------------------------------
def psi(baseline: pd.Series, recent: pd.Series, buckets: int = 10) -> float:
    """
    Compute Population Stability Index (PSI) for a single feature.

    PSI < 0.1  → no drift
    PSI 0.1–0.25 → moderate drift
    PSI > 0.25 → significant drift
    """
    baseline = baseline.dropna()
    recent = recent.dropna()

    if baseline.empty or recent.empty:
        return np.nan

    quantiles = np.linspace(0, 1, buckets + 1)
    cuts = np.unique(baseline.quantile(quantiles))

    if len(cuts) < 2:
        # Not enough unique bins
        return np.nan

    base_counts = np.histogram(baseline, bins=cuts)[0]
    recent_counts = np.histogram(recent, bins=cuts)[0]

    base_dist = base_counts / len(baseline)
    recent_dist = recent_counts / len(recent)

    # Avoid division by zero
    base_dist = np.where(base_dist == 0, 1e-6, base_dist)
    recent_dist = np.where(recent_dist == 0, 1e-6, recent_dist)

    psi_value = np.sum((recent_dist - base_dist) * np.log(recent_dist / base_dist))
    return float(psi_value)


def psi_report(
    baseline: pd.DataFrame,
    recent: pd.DataFrame,
    columns: List[str],
    buckets: int = 10,
) -> Dict[str, float]:
    """
    Compute PSI for multiple columns.

    Returns:
        {column_name: psi_value}
    """
    report = {}
    for col in columns:
        if col not in baseline.columns or col not in recent.columns:
            logger.warning("Column %s missing for PSI", col)
            report[col] = np.nan
            continue

        report[col] = psi(baseline[col], recent[col], buckets=buckets)

    return report
