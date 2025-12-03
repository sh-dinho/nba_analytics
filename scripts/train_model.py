# ============================================================
# File: scripts/train_model.py
# Purpose: Train predictive models on NBA features
# ============================================================

import argparse
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, brier_score_loss, mean_squared_error
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from core.config import TRAINING_FEATURES_FILE, MODEL_FILE_PKL, RESULTS_DIR
from core.log_config import setup_logger
from core.exceptions import DataError
import os

logger = setup_logger("train_model")


def main(target: str = "label", model_type: str = "logistic"):
    """
    Train a model on training features.
    target options:
      - 'label' (binary win/loss)
      - 'margin' (point differential regression)
      - 'outcome_category' (multi-class classification)
    model_type options:
      - 'logistic' (LogisticRegression for binary classification)
      - 'rf' (RandomForestClassifier for classification)
      - 'linear' (LinearRegression for regression)
    """
    if not os.path.exists(TRAINING_FEATURES_FILE):
        raise DataError(f"Training features file not found: {TRAINING_FEATURES_FILE}")

    df = pd.read_csv(TRAINING_FEATURES_FILE)

    if target not in df.columns:
        raise DataError(f"Training data missing '{target}' column")

    # Features: drop non-numeric/categorical identifiers
    feature_cols = [
        c for c in df.columns
        if c not in ["game_id", "home_team", "away_team", "label", "margin", "overtime", "outcome_category"]
    ]
    X = df[feature_cols]
    y = df[target]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Select model
    if target == "margin":
        model = LinearRegression()
    elif target == "outcome_category":
        model = RandomForestClassifier(n_estimators=200, random_state=42)
    else:  # default binary label
        if model_type == "rf":
            model = RandomForestClassifier(n_estimators=200, random_state=42)
        else:
            model = LogisticRegression(max_iter=1000)

    # Fit model
    model.fit(X_train, y_train)

    # Evaluate
    metrics = {}
    if target == "label":
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
        metrics["accuracy"] = accuracy_score(y_test, y_pred)
        metrics["log_loss"] = log_loss(y_test, y_prob)
        metrics["brier"] = brier_score_loss(y_test, y_prob)
        metrics["auc"] = roc_auc_score(y_test, y_prob)
    elif target == "margin":
        y_pred = model.predict(X_test)
        metrics["rmse"] = mean_squared_error(y_test, y_pred, squared=False)
    elif target == "outcome_category":
        y_pred = model.predict(X_test)
        metrics["accuracy"] = accuracy_score(y_test, y_pred)

    # Save model artifact
    artifact = {"model": model, "features": feature_cols, "target": target}
    joblib.dump(artifact, MODEL_FILE_PKL)
    logger.info(f"✅ Model trained on target='{target}' and saved to {MODEL_FILE_PKL}")

    # Save metrics
    os.makedirs(RESULTS_DIR, exist_ok=True)
    metrics_df = pd.DataFrame([metrics])
    metrics_file = os.path.join(RESULTS_DIR, "training_metrics.csv")
    if os.path.exists(metrics_file):
        metrics_df.to_csv(metrics_file, mode="a", header=False, index=False)
    else:
        metrics_df.to_csv(metrics_file, index=False)
    logger.info(f"📊 Metrics saved to {metrics_file}: {metrics}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NBA prediction model")
    parser.add_argument("--target", type=str, default="label",
                        help="Target column: label, margin, outcome_category")
    parser.add_argument("--model_type", type=str, default="logistic",
                        help="Model type: logistic, rf, linear")
    args = parser.parse_args()

    main(target=args.target, model_type=args.model_type)