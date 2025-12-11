# ============================================================
# File: src/model_training/trainer_cli.py
# Purpose: CLI for training models and logging metrics + feature importance
# ============================================================

import logging
import click
import os
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

logger = logging.getLogger("model_training.trainer_cli")
logging.basicConfig(level=logging.INFO)


def evaluate_model(model, X_test, y_test, label):
    """Evaluate model and return metrics dict."""
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


@click.command()
@click.option(
    "--model",
    default="xgb",
    type=click.Choice(["logreg", "rf", "xgb"]),
    help="Model type",
)
@click.option("--season", required=True, type=int, help="Season year (e.g., 2025)")
@click.option("--features", required=True, help="Path to features parquet file")
@click.option(
    "--out", default="models/nba_model.pkl", help="Path to save trained model"
)
@click.option(
    "--metrics_out",
    default="data/results/model_metrics.csv",
    help="Path to save metrics CSV",
)
@click.option(
    "--importance_out",
    default="data/results/feature_importance.csv",
    help="Path to save feature importance CSV",
)
def cli(model, season, features, out, metrics_out, importance_out):
    logger.info("Training model: %s", model)

    # --- Load features ---
    df = pd.read_parquet(features)
    X = df.drop(columns=["WIN"], errors="ignore")
    y = df["WIN"] if "WIN" in df.columns else None

    if y is None:
        logger.error("No WIN column found in features file.")
        return

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- Select model ---
    if model == "logreg":
        clf = LogisticRegression(max_iter=1000)
    elif model == "rf":
        clf = RandomForestClassifier(n_estimators=200, random_state=42)
    else:  # xgb
        clf = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
        )

    # --- Train ---
    clf.fit(X_train, y_train)

    # --- Evaluate ---
    metrics = evaluate_model(clf, X_test, y_test, model)
    os.makedirs(os.path.dirname(metrics_out), exist_ok=True)
    pd.DataFrame([metrics]).to_csv(metrics_out, index=False)
    logger.info("Metrics saved to %s", metrics_out)

    # --- Save model ---
    os.makedirs(os.path.dirname(out), exist_ok=True)
    joblib.dump(clf, out)
    logger.info("Model saved to %s", out)

    # --- Feature importance ---
    importance_df = pd.DataFrame()
    if model == "rf":
        importance_df = pd.DataFrame(
            {"feature": X.columns, "importance": clf.feature_importances_}
        ).sort_values("importance", ascending=False)
    elif model == "xgb":
        importance_df = pd.DataFrame(
            {"feature": X.columns, "importance": clf.feature_importances_}
        ).sort_values("importance", ascending=False)

    if not importance_df.empty:
        importance_df.to_csv(importance_out, index=False)
        logger.info("Feature importance saved to %s", importance_out)


if __name__ == "__main__":
    cli()
