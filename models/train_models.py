# models/train_models.py (Updated with Evaluation and Feature Stats Saving)
import logging
import joblib
import pandas as pd
import json
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import log_loss, mean_squared_error, r2_score, roc_auc_score

from nba_analytics_core.data import fetch_historical_games, engineer_features

def evaluate_models(clf, reg, X_test, y_class_test, y_reg_test) -> dict:
    """Evaluates models and returns a dictionary of metrics."""
    # Classification Metrics
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    y_pred_class = clf.predict(X_test)
    
    metrics = {
        "clf_accuracy": (y_class_test == y_pred_class).mean(),
        "clf_log_loss": log_loss(y_class_test, y_pred_proba),
        "clf_roc_auc": roc_auc_score(y_class_test, y_pred_proba),
    }

    # Regression Metrics
    y_pred_reg = reg.predict(X_test)
    metrics.update({
        "reg_mse": mean_squared_error(y_reg_test, y_pred_reg),
        "reg_r2": r2_score(y_reg_test, y_pred_reg),
    })
    return metrics


def train_models_cached(test_size: float = 0.2, random_state: int = 42):
    df = fetch_historical_games()
    df = engineer_features(df)

    # Drop target columns before splitting
    X = df.drop(["home_win", "total_points"], axis=1)
    y_class = df["home_win"]
    y_reg = df["total_points"]

    # Implement Train/Test Split
    X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
        X, y_class, y_reg, test_size=test_size, random_state=random_state
    )

    # Train Models on Training Data
    clf = LogisticRegression(max_iter=1000).fit(X_train, y_class_train)
    reg = LinearRegression().fit(X_train, y_reg_train)

    # --- IMPROVEMENT: Save Feature Statistics from Training Data ---
    feature_stats = {
        'mean': X_train.mean().to_dict(),
        'std': X_train.std().to_dict(),
    }
    with open("artifacts/feature_stats.json", "w") as f:
        json.dump(feature_stats, f, indent=4)
    logging.info("Historical feature statistics saved.")
    
    # Evaluate Models and Save Metrics
    metrics = evaluate_models(clf, reg, X_test, y_class_test, y_reg_test)
    with open("artifacts/model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    logging.info("Model Metrics saved to artifacts/model_metrics.json")
    logging.info(f"Classifier AUC: {metrics['clf_roc_auc']:.4f}, Regressor R2: {metrics['reg_r2']:.4f}")
    
    # Save Models
    joblib.dump(clf, "models/classification_model.pkl")
    joblib.dump(reg, "models/regression_model.pkl")

    logging.info("✔ Models trained and saved")

if __name__ == "__main__":
    train_models_cached()