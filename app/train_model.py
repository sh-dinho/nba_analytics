# ============================================================
# File: app/train_model.py
# Purpose: Train NBA game predictor model (logistic, XGB, NN) and save artifact
# ============================================================

from pathlib import Path
import argparse
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, brier_score_loss

from core.config import (
    MODELS_DIR,
    MODEL_FILE_PKL,
    MODEL_FILE_H5,
    TRAINING_FEATURES_FILE,
    ensure_dirs,
)
from core.log_config import setup_logger
from core.exceptions import PipelineError

# Optional imports for other models
try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
except ImportError:
    tf = None

logger = setup_logger("train_model")

# Ensure models directory exists
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _load_data() -> pd.DataFrame:
    """
    Load training data. Prefer TRAINING_FEATURES_FILE; fallback to a small synthetic dataset.
    """
    if TRAINING_FEATURES_FILE.exists():
        logger.info(f"📄 Loading training data from {TRAINING_FEATURES_FILE}")
        return pd.read_csv(TRAINING_FEATURES_FILE)

    logger.warning("⚠️ TRAINING_FEATURES_FILE not found. Using synthetic demo data.")
    return pd.DataFrame({
        "feature1": [0.1, 0.2, 0.8, 0.9, 0.4, 0.6],
        "feature2": [1, 2, 3, 4, 5, 6],
        "label":    [0, 0, 1, 1, 0, 1]
    })


def train_logistic(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def train_xgb(X_train, y_train):
    if xgb is None:
        raise ImportError("XGBoost not installed in this environment")
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_estimators=200)
    model.fit(X_train, y_train)
    return model


def train_nn(X_train, y_train, input_dim):
    if tf is None:
        raise ImportError("TensorFlow not installed in this environment")
    model = Sequential([
        Dense(16, activation="relu", input_dim=input_dim),
        Dense(8, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=20, batch_size=8, verbose=0)
    return model


def _compute_metrics_binary(y_true, y_pred, y_proba=None):
    metrics = {}
    try:
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
    except Exception:
        metrics["accuracy"] = None
    if y_proba is not None:
        try:
            metrics["log_loss"] = log_loss(y_true, y_proba)
        except Exception:
            metrics["log_loss"] = None
        try:
            metrics["brier"] = brier_score_loss(y_true, y_proba)
        except Exception:
            metrics["brier"] = None
        try:
            metrics["auc"] = roc_auc_score(y_true, y_proba)
        except Exception:
            metrics["auc"] = None
    else:
        metrics["log_loss"] = None
        metrics["brier"] = None
        metrics["auc"] = None
    return metrics


def main(model_type: str | None = None):
    """
    Train a model and save the artifact. Returns a metrics dict.

    If called via CLI, parses --model_type; if called programmatically (from setup_all),
    you can pass model_type directly.
    """
    try:
        ensure_dirs()
        # CLI parsing only if not provided programmatically
        if model_type is None:
            parser = argparse.ArgumentParser()
            parser.add_argument("--model_type", type=str, default="logistic",
                                choices=["logistic", "xgb", "nn"],
                                help="Type of model to train")
            args = parser.parse_args()
            model_type = args.model_type

        # Load data
        data = _load_data()
        # Heuristic: use all numeric columns except 'label' as features
        if "label" not in data.columns:
            raise PipelineError("Training data must contain a 'label' column.")
        feature_cols = [c for c in data.select_dtypes(include="number").columns if c != "label"]
        if not feature_cols:
            raise PipelineError("No numeric feature columns found for training.")

        X = data[feature_cols]
        y = data["label"]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Train selected model
        if model_type == "logistic":
            model = train_logistic(X_train, y_train)
            y_pred = model.predict(X_test)
            # Probability for metrics
            try:
                y_proba = model.predict_proba(X_test)[:, 1]
            except Exception:
                y_proba = None
            metrics = _compute_metrics_binary(y_test, y_pred, y_proba)
            joblib.dump(model, MODEL_FILE_PKL)
            logger.info(f"📦 Logistic model saved to {MODEL_FILE_PKL}")

        elif model_type == "xgb":
            model = train_xgb(X_train, y_train)
            y_pred = model.predict(X_test)
            # xgb.predict may return class labels; proba via predict_proba
            try:
                y_proba = model.predict_proba(X_test)[:, 1]
            except Exception:
                y_proba = None
            metrics = _compute_metrics_binary(y_test, y_pred, y_proba)
            joblib.dump(model, MODEL_FILE_PKL)
            logger.info(f"📦 XGBoost model saved to {MODEL_FILE_PKL}")

        elif model_type == "nn":
            model = train_nn(X_train, y_train, input_dim=X_train.shape[1])
            # NN predicted probabilities
            y_proba = model.predict(X_test, verbose=0).ravel()
            y_pred = (y_proba >= 0.5).astype(int)
            metrics = _compute_metrics_binary(y_test, y_pred, y_proba)
            model.save(MODEL_FILE_H5)
            logger.info(f"📦 NN model saved to {MODEL_FILE_H5}")

        else:
            raise PipelineError(f"Unknown model_type: {model_type}")

        # Log headline metric
        acc = metrics.get("accuracy")
        if acc is not None:
            logger.info(f"✅ {model_type.upper()} Accuracy: {acc:.3f}")
        else:
            logger.info(f"✅ {model_type.upper()} training finished (accuracy unavailable)")

        logger.info("📦 Training complete. Model artifact saved.")
        return metrics

    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise PipelineError(f"Model training failed: {e}")


if __name__ == "__main__":
    main()