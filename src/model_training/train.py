# ============================================================
# File: src/model_training/train.py
# Purpose: Unified training for Logistic Regression and XGBoost
# Project: nba_analysis
# Version: 1.6 (adds config, MLflow, SHAP, flexible cache format)
# ============================================================

import logging
import os
import joblib
import pandas as pd
import yaml
import mlflow

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from src.interpretability.shap_analysis import run_shap
from mlflow_setup import configure_mlflow, start_run_with_metadata


def train_model(cache_file, model_type="logreg", out_dir="models", run_shap=False, config_file="config.yaml"):
    """
    Train either Logistic Regression or XGBoost model.
    Args:
        cache_file (str): Path to cached dataset (CSV or Parquet).
        model_type (str): 'logreg' or 'xgb'.
        out_dir (str): Directory to save trained models.
        run_shap (bool): Whether to run SHAP analysis after training (XGB only).
        config_file (str): Path to config.yaml for MLflow settings.
    Returns:
        dict: {"model_path": str, "accuracy": float, "logloss": float}
    """
    # Load dataset
    if cache_file.endswith(".parquet"):
        df = pd.read_parquet(cache_file)
    elif cache_file.endswith(".csv"):
        df = pd.read_csv(cache_file)
    else:
        raise ValueError(f"Unsupported cache file format: {cache_file}")

    if "win" not in df.columns:
        logging.error("Target column 'win' missing")
        return None

    X = df.drop(columns=["win"])
    y = df["win"]

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="mean"), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ]
    )

    if model_type == "logreg":
        clf = LogisticRegression(max_iter=1000)
        model_file = "logreg.pkl"
    elif model_type == "xgb":
        pos = int(y.sum())
        neg = int(len(y) - pos)
        scale_pos_weight = (neg / pos) if pos > 0 else 1.0
        clf = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            n_jobs=4,
        )
        model_file = "nba_xgb.pkl"
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    pipeline = Pipeline([("preprocessor", preprocessor), ("clf", clf)])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(set(y)) > 1 else None
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_proba)

    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, model_file)
    joblib.dump(pipeline, model_path)

    logging.info(f"{model_type} trained. Accuracy: {acc:.3f}, Logloss: {loss:.3f}")

    # Load config for MLflow
    if os.path.exists(config_file):
        with open(config_file) as f:
            cfg = yaml.safe_load(f)
        mlflow_cfg = cfg.get("mlflow", {})
    else:
        mlflow_cfg = {}

    if mlflow_cfg.get("enabled", False):
        configure_mlflow(experiment_name=mlflow_cfg.get("experiment", "nba_training"))
        with start_run_with_metadata(f"train_{model_type}"):
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("cache_file", cache_file)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("logloss", loss)
            mlflow.log_artifact(model_path, artifact_path="models")

    # Optional SHAP analysis
    if run_shap and model_type == "xgb":
        shap_dir = os.path.join(out_dir, "interpretability")
        os.makedirs(shap_dir, exist_ok=True)
        logging.info("Running SHAP analysis...")
        run_shap(model_path, cache_file, out_dir=shap_dir, top_n=5)

    return {"model_path": model_path, "accuracy": acc, "logloss": loss}
