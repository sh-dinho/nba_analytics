# ============================================================
# File: src/model_training/training.py
# Purpose: Train logistic regression model and log results with MLflow
# Project: nba_analysis
# Version: 1.3 (adds preprocessing, stratification, logloss, local save)
# ============================================================

import mlflow
import mlflow.sklearn
import pandas as pd
import os, joblib, logging
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from src.utils.io import load_dataframe

def train_logreg(features_path: str, out_dir: str, test_size: float = 0.2, random_state: int = 42):
    """
    Train a logistic regression model using features stored in a parquet file.
    Automatically handles numeric/categorical preprocessing and logs metrics to MLflow.
    """
    # Load features
    df = load_dataframe(features_path)
    if "win" not in df.columns:
        raise ValueError("Training data must include a 'win' column as target label")

    # Split into features and target
    X = df.drop(columns=["win"])
    y = df["win"]

    if X.empty or y.empty:
        raise ValueError("Training data is empty")

    # Separate numeric and categorical features
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="mean"), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ]
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    # Train/test split with stratification
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if len(set(y)) > 1 else None
    )

    # Train model
    pipeline.fit(X_train, y_train)

    # Evaluate
    preds = pipeline.predict(X_val)
    proba = pipeline.predict_proba(X_val)
    metrics = {
        "accuracy": accuracy_score(y_val, preds),
        "precision": precision_score(y_val, preds, zero_division=0),
        "recall": recall_score(y_val, preds, zero_division=0),
        "f1_score": f1_score(y_val, preds, zero_division=0),
        "logloss": log_loss(y_val, proba),
    }

    # Save locally
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "logreg.pkl")
    joblib.dump(pipeline, model_path)
    logging.info(f"Model saved to {model_path}")

    # Log to MLflow
    with mlflow.start_run():
        mlflow.log_param("max_iter", 1000)
        mlflow.log_param("solver", pipeline.named_steps["clf"].solver)
        mlflow.log_param("C", pipeline.named_steps["clf"].C)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        mlflow.sklearn.log_model(pipeline, "logreg_model")

    return pipeline, metrics
