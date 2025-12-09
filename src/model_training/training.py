# ============================================================
# Path: src/model_training/training.py
# File: training.py
# Purpose: Train logistic regression model and log results with MLflow
# Project: nba_analysis
# ============================================================

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from src.utils.io import load_dataframe


def train_logreg(features_path: str, out_dir: str, test_size: float = 0.2, random_state: int = 42):
    """
    Train a logistic regression model using features stored in a parquet file.
    Logs metrics and model artifacts to MLflow.

    Parameters
    ----------
    features_path : str
        Path to the parquet file containing features and target column 'win'.
    out_dir : str
        Directory to save model artifacts.
    test_size : float
        Fraction of data to use for validation.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    model : LogisticRegression
        Trained logistic regression model.
    metrics : dict
        Dictionary of evaluation metrics.
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

    # Train/test split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Train model
    model = LogisticRegression(max_iter=1000)
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        raise RuntimeError(f"Model training failed: {e}")

    # Evaluate
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    precision = precision_score(y_val, preds, zero_division=0)
    recall = recall_score(y_val, preds, zero_division=0)
    f1 = f1_score(y_val, preds, zero_division=0)

    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

    # Log to MLflow
    with mlflow.start_run():
        mlflow.log_param("max_iter", 1000)
        mlflow.log_param("solver", model.solver)
        mlflow.log_param("C", model.C)

        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        mlflow.sklearn.log_model(model, "logreg_model")

    return model, metrics
