# ============================================================
# File: app/train_model.py
# Purpose: Train NBA game prediction model and save artifacts
# ============================================================

import os
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, roc_auc_score

from core.config import TRAINING_FEATURES_FILE, MODEL_FILE_PKL
from core.log_config import setup_logger
from core.exceptions import PipelineError, DataError
from core.utils import ensure_columns

logger = setup_logger("train_model")


def main() -> dict:
    """Train a logistic regression model on training features and save model + feature order."""
    if not os.path.exists(TRAINING_FEATURES_FILE):
        raise FileNotFoundError(f"{TRAINING_FEATURES_FILE} not found. Run build_features first.")

    try:
        df = pd.read_csv(TRAINING_FEATURES_FILE)
    except Exception as e:
        raise PipelineError(f"Failed to load training data: {e}")

    # Ensure label column exists
    if "label" not in df.columns:
        raise DataError("Training data missing 'label' column")

    # Separate features and target
    y = df["label"]
    X = df.drop(columns=["label"])
    feature_order = list(X.columns)

    # Train logistic regression
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    # Predictions for metrics
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    metrics = {}
    try:
        metrics["accuracy"] = accuracy_score(y, y_pred)
    except Exception:
        metrics["accuracy"] = None
    try:
        metrics["log_loss"] = log_loss(y, y_proba)
    except Exception:
        metrics["log_loss"] = None
    try:
        metrics["brier"] = brier_score_loss(y, y_proba)
    except Exception:
        metrics["brier"] = None
    try:
        metrics["auc"] = roc_auc_score(y, y_proba)
    except Exception:
        metrics["auc"] = None

    # ✅ Save both model and feature order in a dict
    os.makedirs(os.path.dirname(MODEL_FILE_PKL), exist_ok=True)
    joblib.dump({"model": model, "features": feature_order}, MODEL_FILE_PKL)
    logger.info(f"📦 Logistic model + features saved to {MODEL_FILE_PKL}")

    # Log headline metrics
    logger.info("=== TRAINING METRICS ===")
    for k, v in metrics.items():
        if v is not None:
            logger.info(f"{k.capitalize()}: {v:.3f}")
        else:
            logger.info(f"{k.capitalize()}: unavailable")

    return metrics


if __name__ == "__main__":
    main()