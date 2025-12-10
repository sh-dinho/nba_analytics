# ============================================================
# File: src/interpretability/shap_analysis.py
# Purpose: SHAP interpretability for XGBoost model
# ============================================================

import shap, joblib, os, pandas as pd, matplotlib.pyplot as plt
import mlflow
from datetime import datetime

def run_shap(model_path, cache_file, out_dir="results/interpretability", top_n=3):
    os.makedirs(out_dir, exist_ok=True)
    pipeline = joblib.load(model_path)
    xgb_model = pipeline.named_steps["clf"]
    df = pd.read_parquet(cache_file)
    X = df.drop(columns=["target"], errors="ignore")

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(X)

    summary_path = os.path.join(out_dir, "shap_summary.png")
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig(summary_path, bbox_inches="tight")
    plt.close()

    mean_abs = pd.Series(shap_values.values.mean(axis=0).abs(), index=X.columns).sort_values(ascending=False)
    top_features = mean_abs.index[:top_n].tolist()
    dep_paths = []

    for feat in top_features:
        dep_path = os.path.join(out_dir, f"shap_dependence_{feat}.png")
        shap.dependence_plot(feat, shap_values.values, X, show=False)
        plt.savefig(dep_path, bbox_inches="tight")
        plt.close()
        dep_paths.append(dep_path)

    # Log to MLflow
    with mlflow.start_run(run_name="shap_analysis"):
        mlflow.log_artifact(summary_path, artifact_path="interpretability")
        for p in dep_paths:
            mlflow.log_artifact(p, artifact_path="interpretability")

    print(f"✅ SHAP report logged. Top features: {top_features}")
