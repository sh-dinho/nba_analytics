# ============================================================
# Path: src/interpretability/shap_analysis.py
# Purpose: Run SHAP analysis on trained XGBoost model and log plots to MLflow
# Version: 1.1 (interpretability upgrade)
# ============================================================

import shap
import xgboost as xgb
import pandas as pd
import mlflow
import matplotlib.pyplot as plt

def run_shap(model_path, cache_file="data/cache/features_full.parquet"):
    """
    Generate SHAP summary plot and log artifact to MLflow.
    """
    # Load model
    model = xgb.Booster()
    model.load_model(model_path)

    # Load features
    df = pd.read_parquet(cache_file)
    X = df.drop(columns=["win"])

    # Run SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Plot summary
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig("shap_summary.png", bbox_inches="tight")

    # Log artifact to MLflow
    with mlflow.start_run(run_name="shap_analysis_v1.1"):
        mlflow.log_artifact("shap_summary.png", artifact_path="interpretability")

    print("✅ SHAP summary plot logged to MLflow")
