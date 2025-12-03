# ============================================================
# File: app/train_model.py
# Purpose: Train NBA game prediction model with numeric + categorical features
# ============================================================

import os
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, roc_auc_score

from core.config import TRAINING_FEATURES_FILE, MODEL_FILE_PKL
from core.log_config import setup_logger
from core.utils import ensure_columns

logger = setup_logger("train_model")


def main() -> dict:
    """
    Train a logistic regression model on historical training features.
    Handles both numeric and categorical features.
    Saves model + feature names in a consistent dict structure.
    Returns training metrics.
    """
    if not os.path.exists(TRAINING_FEATURES_FILE):
        raise FileNotFoundError(f"{TRAINING_FEATURES_FILE} not found. Run build_features_for_training first.")

    df = pd.read_csv(TRAINING_FEATURES_FILE)

    # Define feature sets
    numeric_features = [
        "home_avg_pts", "home_avg_ast", "home_avg_reb", "home_avg_games_played",
        "away_avg_pts", "away_avg_ast", "away_avg_reb", "away_avg_games_played"
    ]
    categorical_features = ["home_team", "away_team"]

    ensure_columns(df, numeric_features + categorical_features + ["label"], "training features")

    X = df[numeric_features + categorical_features]
    y = df["label"]

    # Preprocessor: scale numeric, one-hot encode categorical
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    # Build pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X, y)

    # Predictions for metrics
    y_pred = pipeline.predict(X)
    y_prob = pipeline.predict_proba(X)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "log_loss": log_loss(y, y_prob),
        "brier": brier_score_loss(y, y_prob),
        "auc": roc_auc_score(y, y_prob)
    }

    # ✅ Save as dict with model + feature lists
    artifact = {
        "model": pipeline,
        "features": numeric_features + categorical_features,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features
    }
    joblib.dump(artifact, MODEL_FILE_PKL)

    logger.info(f"📦 Logistic model (numeric + categorical) saved to {MODEL_FILE_PKL}")
    logger.info("=== TRAINING METRICS ===")
    for k, v in metrics.items():
        logger.info(f"{k.capitalize()}: {v:.3f}")

    return metrics


if __name__ == "__main__":
    main()