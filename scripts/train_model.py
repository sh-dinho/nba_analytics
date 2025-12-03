# ============================================================
# File: app/train_model.py
# Purpose: Train NBA game prediction model and save artifacts
# ============================================================

import os
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from core.config import TRAINING_FEATURES_FILE, MODEL_FILE_PKL
from core.log_config import setup_logger
from core.exceptions import PipelineError, DataError

logger = setup_logger("train_model")


def main() -> dict:
    """Train a logistic regression model with numeric + categorical features."""
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

    # Identify numeric and categorical columns
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    logger.info(f"📊 Training with {len(numeric_cols)} numeric and {len(categorical_cols)} categorical features")

    # Preprocessor: passthrough numeric, one-hot encode categorical
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    # Pipeline: preprocessing + logistic regression
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    # Fit model
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
    joblib.dump({"model": model, "features": numeric_cols + categorical_cols}, MODEL_FILE_PKL)
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