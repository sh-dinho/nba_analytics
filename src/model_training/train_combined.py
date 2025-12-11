# ============================================================
# File: src/model_training/train_combined.py
# Purpose: Unified training interface for LogReg or XGB
# Project: nba_analysis
# Version: 2.0 (config-driven, MLflow, SHAP, metrics return)
# ============================================================

import os
import logging
import yaml
import mlflow

from src.model_training.train_logreg import train_logreg
from src.model_training.train_xgb import train_xgb
from mlflow_setup import configure_mlflow, start_run_with_metadata


def train_model(cache_file, out_dir="models", model_type="logreg",
                run_shap=False, config_file="config.yaml"):
    """
    Unified training interface for Logistic Regression or XGBoost.
    Args:
        cache_file (str): Path to cached dataset (CSV or Parquet).
        out_dir (str): Directory to save trained models.
        model_type (str): 'logreg' or 'xgb'.
        run_shap (bool): Whether to run SHAP analysis after training (XGB only).
        config_file (str): Path to config.yaml for MLflow settings.
    Returns:
        dict: {"model_path": str, "metrics": dict}
    """
    os.makedirs(out_dir, exist_ok=True)

    # Load config
    if os.path.exists(config_file):
        with open(config_file) as f:
            cfg = yaml.safe_load(f)
        mlflow_cfg = cfg.get("mlflow", {})
    else:
        mlflow_cfg = {}

    # Configure MLflow
    if mlflow_cfg.get("enabled", False):
        configure_mlflow(experiment_name=mlflow_cfg.get("experiment", "nba_training"))

    logging.info(f"Starting training: model_type={model_type}, cache_file={cache_file}")

    # Train model
    if model_type == "logreg":
        result = train_logreg(cache_file, out_dir, config_file=config_file)
    elif model_type == "xgb":
        result = train_xgb(cache_file, out_dir, run_shap=run_shap, config_file=config_file)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model_path = result["model_path"]
    metrics = result.get("metrics", {})

    # Log to MLflow
    if mlflow_cfg.get("enabled", False):
        with start_run_with_metadata(f"train_{model_type}"):
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("cache_file", cache_file)
            for k, v in metrics.items():
                mlflow.log_metric(k, v)
            mlflow.log_artifact(model_path, artifact_path="models")

    logging.info(f"✅ Training complete. Model saved to {model_path}. Metrics: {metrics}")
    return {"model_path": model_path, "metrics": metrics}
