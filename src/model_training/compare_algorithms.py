# ============================================================
# File: src/model_training/compare_algorithms.py
# Purpose: Compare multiple ML algorithms and log metrics
# ============================================================

import logging
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from src.schemas import normalize_features

logger = logging.getLogger("model_training.compare_algorithms")
logging.basicConfig(level=logging.INFO)


def evaluate_model(model, X_test, y_test, label):
    y_pred = model.predict(X_test)
    y_prob = None
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
    except Exception:
        pass

    metrics = {
        "model": label,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob) if y_prob is not None else None,
    }
    return metrics


def main():
    features_path = "data/cache/features_full.parquet"
    if not os.path.exists(features_path):
        logger.error("Features file not found at %s. Aborting.", features_path)
        return

    df = pd.read_parquet(features_path)
    df = normalize_features(df)

    if df.empty or "WIN" not in df.columns:
        logger.error("Features file is empty or missing WIN column. Aborting training.")
        return

    # Drop non-numeric columns if necessary (basic guard)
    non_feature_cols = ["GAME_ID", "WL"]
    X = df.drop(columns=[c for c in non_feature_cols if c in df.columns] + ["WIN"])
    y = df["WIN"]

    if y.isnull().all():
        logger.error("WIN column has no values. Aborting training.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X.fillna(0), y.fillna(0), test_size=0.2, random_state=42
    )

    results = []

    logger.info("Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    results.append(evaluate_model(lr, X_test, y_test, "LogisticRegression"))

    logger.info("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    results.append(evaluate_model(rf, X_test, y_test, "RandomForest"))

    logger.info("Training XGBoost...")
    xgb = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
    )
    xgb.fit(X_train, y_train)
    results.append(evaluate_model(xgb, X_test, y_test, "XGBoost"))

    out_path = "data/results/model_comparison.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pd.DataFrame(results).to_csv(out_path, index=False)
    logger.info("Comparison metrics saved to %s", out_path)


if __name__ == "__main__":
    main()
