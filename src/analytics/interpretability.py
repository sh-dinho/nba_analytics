# ============================================================
# File: src/analytics/interpretability.py
# Purpose: SHAP analysis + algorithm comparison
# Project: nba_analysis
# Version: 1.0 (merged interpretability)
# ============================================================

import shap
import pandas as pd
from src.utils.common import configure_logging

# Set up logging for this module
logger = configure_logging(name="analytics.interpretability")


def shap_analysis(model, X: pd.DataFrame):
    """
    Run SHAP analysis on a trained model to explain predictions.

    Args:
        model: The trained machine learning model to explain.
        X (pd.DataFrame): The features for which SHAP values will be computed.

    Returns:
        shap_values: SHAP values computed for the input data.
    """
    # Create a SHAP Explainer for the model
    explainer = shap.Explainer(model, X)

    # Calculate SHAP values
    shap_values = explainer(X)

    logger.info("SHAP analysis complete.")
    return shap_values


def compare_algorithms(results: dict):
    """
    Compare multiple machine learning algorithms based on performance metrics.

    Args:
        results (dict): Dictionary where each key is the algorithm name, and the value
                        is a dictionary containing metrics such as accuracy, precision, etc.

    Returns:
        pd.DataFrame: A DataFrame sorted by the chosen performance metric (e.g., accuracy).
    """
    # Convert the results into a DataFrame
    df = pd.DataFrame(results).T

    # Log completion of comparison
    logger.info("Algorithm comparison complete.")

    # Return the sorted DataFrame based on accuracy (descending)
    return df.sort_values("accuracy", ascending=False)
