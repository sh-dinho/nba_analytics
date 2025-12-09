# ============================================================
# Path: src/model_training/train_logreg.py
# Filename: train_logreg.py
# Author: Your Team
# Date: December 9, 2025
# Purpose: Train and log NBA logistic regression model
# ============================================================

import pandas as pd
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import mlflow

def train_logreg(features_path: str, out_dir: str = "models") -> dict:
    """
    Train a logistic regression model on NBA features and log metrics.

    Args:
        features_path (str): Path to parquet file with features.
        out_dir (str): Directory to save model.

    Returns:
        dict: Training metrics and model path.
    """
    # Load features
    df = pd.read_parquet(features_path)
    X = df.drop(columns=["win"])
    y = df["win"]

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    # Evaluate
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds)

    # Save model
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "nba_logreg.pkl")
    joblib.dump(model, model_path)

    # Log metrics with MLflow
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    return {
        "metrics": {"accuracy": acc, "f1_score": f1},
        "model_path": model_path
    }
