# ============================================================
# File: src/interpretability/shap_analysis.py
# Purpose: SHAP interpretability for XGBoost model
# Version: 1.3 (numeric coercion, safer SHAP calls, metrics logging)
# ============================================================

import os
import logging
import joblib
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import shap
import numpy as np

logger = logging.getLogger("interpretability.shap")


def run_shap(
    model_path, cache_file, out_dir="results/interpretability", top_n=3, clf_step="clf"
):
    """
    Run SHAP analysis for an XGBoost model inside a sklearn pipeline.
    Args:
        model_path (str): Path to saved pipeline (joblib).
        cache_file (str): Path to cached dataset (parquet or csv).
        out_dir (str): Directory to save SHAP plots.
        top_n (int): Number of top features for dependence plots.
        clf_step (str): Name of classifier step in pipeline.
    """
    os.makedirs(out_dir, exist_ok=True)

    pipeline = joblib.load(model_path)
    if clf_step not in pipeline.named_steps:
        raise KeyError(
            f"Classifier step '{clf_step}' not found in pipeline steps: {list(pipeline.named_steps.keys())}"
        )
    xgb_model = pipeline.named_steps[clf_step]

    # Load dataset
    if cache_file.endswith(".parquet"):
        df = pd.read_parquet(cache_file)
    elif cache_file.endswith(".csv"):
        df = pd.read_csv(cache_file)
    else:
        raise ValueError(f"Unsupported cache file format: {cache_file}")

    # Drop target column(s) and coerce all features to numeric
    X = df.drop(columns=["target", "TARGET"], errors="ignore")
    X = X.apply(pd.to_numeric, errors="coerce")

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(X)

    # Summary plot
    summary_path = os.path.join(out_dir, "shap_summary.png")
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig(summary_path, bbox_inches="tight")
    plt.close()

    # Top features by mean absolute SHAP
    mean_abs = pd.Series(
        np.abs(np.asarray(shap_values.values)).mean(axis=0), index=X.columns
    ).sort_values(ascending=False)
    top_features = mean_abs.index[:top_n].tolist()
    dep_paths = []

    for feat in top_features:
        dep_path = os.path.join(out_dir, f"shap_dependence_{feat}.png")
        shap.dependence_plot(feat, shap_values, X, show=False)
        plt.savefig(dep_path, bbox_inches="tight")
        plt.close()
        dep_paths.append(dep_path)

    # Log to MLflow (nested run)
    with mlflow.start_run(run_name="shap_analysis", nested=True):
        mlflow.log_artifact(summary_path, artifact_path="interpretability")
        for p in dep_paths:
            mlflow.log_artifact(p, artifact_path="interpretability")
        # Log mean SHAP values for top features
        for feat in top_features:
            mlflow.log_metric(f"shap_mean_abs_{feat}", float(mean_abs[feat]))

    logger.info("✅ SHAP report logged. Top features: %s", top_features)
