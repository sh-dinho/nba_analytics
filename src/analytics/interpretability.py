# ============================================================
# File: src/analytics/interpretability.py
# Purpose: SHAP analysis + algorithm comparison
# Project: nba_analysis
# Version: 1.0 (merged interpretability)
# ============================================================

import shap
import pandas as pd
from src.utils.common import configure_logging

logger = configure_logging(name="analytics.interpretability")


def shap_analysis(model, X: pd.DataFrame):
    """Run SHAP analysis on a trained model."""
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    logger.info("SHAP analysis complete.")
    return shap_values


def compare_algorithms(results: dict):
    """Compare algorithms based on accuracy or other metrics."""
    df = pd.DataFrame(results).T
    logger.info("Algorithm comparison complete.")
    return df.sort_values("accuracy", ascending=False)
