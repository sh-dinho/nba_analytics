# ============================================================
# File: src/monitoring/drift.py
# Purpose: Data drift detection utilities (KS-test, PSI)
# Version: 3.0
# Author: Your Team
# Date: December 2025
# ============================================================

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from loguru import logger

from src.config import MONITORING

DEFAULT_ALPHA = MONITORING["drift_alpha"]
MIN_SAMPLES = MONITORING["min_samples"]


def ks_drift_report(
    baseline: pd.DataFrame,
    recent: pd.DataFrame,
    columns: List[str],
    alpha: float = DEFAULT_ALPHA,
    min_samples: int = MIN_SAMPLES,
) -> Dict[str, Dict[str, float]]:
    report: Dict[str, Dict[str, float]] = {}
    for col in columns:
        if col not in baseline.columns or col not in recent.columns:
            logger.warning("Column %s missing in baseline or recent", col)
            report[col] = {"statistic": np.nan, "pvalue": np.nan, "drift": np.nan}
            continue
        b = baseline[col].dropna().values
        r = recent[col].dropna().values
        if len(b) < min_samples or len(r) < min_samples:
            logger.warning(
                "Insufficient samples for KS on %s (baseline=%d, recent=%d)",
                col,
                len(b),
                len(r),
            )
            report[col] = {"statistic": np.nan, "pvalue": np.nan, "drift": np.nan}
            continue
        stat, pval = ks_2samp(b, r)
        report[col] = {
            "statistic": float(stat),
            "pvalue": float(pval),
            "drift": float(pval < alpha),
        }
    return report


def summarize_drift(report: Dict[str, Dict[str, float]]) -> Tuple[int, int]:
    tested = [c for c, v in report.items() if not np.isnan(v.get("pvalue", np.nan))]
    drifted = [c for c in tested if report[c]["drift"] == 1.0]
    return len(drifted), len(tested)
