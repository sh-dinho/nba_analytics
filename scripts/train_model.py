# File: scripts/train_model.py

import os
import pandas as pd
import joblib
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, roc_auc_score
from scripts.utils import setup_logger

logger = setup_logger("train_model")

DATA_DIR = "data"
MODELS_DIR = "models"
RESULTS_DIR = "results"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

METRICS_LOG = os.path.join(RESULTS_DIR, "training_metrics.csv")

def main():
    features_file = os.path.join(DATA_DIR, "training_features.csv")
    model_file = os.path.join(MODELS_DIR, "game_predictor.pkl")

    if not os.path.exists(features_file):
        logger.error(f"{features_file} not found. Build features first.")
        raise FileNotFoundError(features_file)

    logger.info("Training model...")
    df = pd.read_csv(features_file)
    if df.empty:
        logger.error("Training features CSV is empty.")
        raise ValueError("No training data available.")

    # ✅ Exclude identifiers and target
    drop_cols = ["home_win", "game_id", "home_team", "away_team"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].fillna(0)
    y = df["home_win"].fillna(0).astype(int)

    if len(df) < 10:
        logger.warning("Small dataset. Training on all data without split.")
        clf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
        clf.fit(X, y)
        y_pred = clf.predict(X)
        y_prob = clf.predict_proba(X)[:, 1]
        metrics = {"accuracy": accuracy_score(y, y_pred)}
        if len(set(y)) > 1:
            metrics["log_loss"] = log_loss(y, y_prob)
            metrics["brier"] = brier_score_loss(y, y_prob)
            metrics["auc"] = roc_auc_score(y, y_prob)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if len(set(y)) > 1 else None
        )
        clf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]
        metrics = {"accuracy": accuracy_score(y_test, y_pred)}
        if len(set(y_test)) > 1:
            metrics["log_loss"] = log_loss(y_test, y_prob)
            metrics["brier"] = brier_score_loss(y_test, y_prob)
            metrics["auc"] = roc_auc_score(y_test, y_prob)

    joblib.dump(clf, model_file)
    logger.info(f"Model saved to {model_file}")

    logger.info("=== TRAINING METRICS ===")
    for k, v in metrics.items():
        logger.info(f"{k}: {v:.4f}")

    run_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metrics_entry = pd.DataFrame([{
        "timestamp": run_time,
        "model_file": model_file,
        "n_features": len(feature_cols),
        "n_samples": len(df),
        **metrics
    }])
    metrics_entry.to_csv(
        METRICS_LOG,
        mode="a" if os.path.exists(METRICS_LOG) else "w",
        index=False,
        header=not os.path.exists(METRICS_LOG),
    )
    logger.info(f"Training metrics appended to {METRICS_LOG}")
    return metrics

if __name__ == "__main__":
    main()