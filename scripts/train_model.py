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
    
    # 🌟 FIX: Save the model to a fixed path, which the prediction script expects
    model_file = os.path.join(MODELS_DIR, "game_predictor.pkl")

    if not os.path.exists(features_file):
        logger.error(f"{features_file} not found. Build features first.")
        raise FileNotFoundError(features_file)

    logger.info("Training model...")

    df = pd.read_csv(features_file)

    if df.empty:
        logger.error("Training features CSV is empty!")
        raise ValueError("No training data available.")

    # Auto-select numeric columns except target
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if "home_win" not in numeric_cols:
        raise KeyError("Target column 'home_win' not found in features CSV.")
    feature_cols = [c for c in numeric_cols if c != "home_win"]

    X = df[feature_cols]
    y = df["home_win"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "log_loss": log_loss(y_test, y_prob),
        "brier": brier_score_loss(y_test, y_prob),
        "auc": roc_auc_score(y_test, y_prob)
    }

    joblib.dump(clf, model_file)
    logger.info(f"✅ Model saved to {model_file}")

    # Log metrics
    logger.info("=== TRAINING METRICS ===")
    for k, v in metrics.items():
        logger.info(f"{k}: {v:.4f}")

    run_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metrics_entry = pd.DataFrame([{
        "timestamp": run_time,
        "model_file": model_file,
        "n_features": len(feature_cols),
        "n_samples": len(df),
        "accuracy": metrics["accuracy"],
        "log_loss": metrics["log_loss"],
        "auc": metrics["auc"]
    }])

    if os.path.exists(METRICS_LOG):
        metrics_entry.to_csv(METRICS_LOG, mode="a", header=False, index=False)
    else:
        metrics_entry.to_csv(METRICS_LOG, index=False)
    logger.info(f"📁 Training metrics appended to {METRICS_LOG}")
    
    return metrics

if __name__ == "__main__":
    main()