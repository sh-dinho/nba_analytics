#!/usr/bin/env python
# ============================================================
# File: src/model_training/trainer_cli.py
# Purpose: Train NBA prediction models with preprocessing + evaluation
# ============================================================

import argparse
import logging
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
import joblib

from src.utils.logging_config import configure_logging


def preprocess_features(df: pd.DataFrame, logger) -> (pd.DataFrame, pd.Series):
    """Prepare X and y for training with numeric features."""

    if "win" not in df.columns:
        raise ValueError("Target column 'win' not found in features file.")
    y = df["win"]

    # Drop identifiers
    X = df.drop(columns=["win", "GAME_ID"], errors="ignore")

    # Encode categorical columns
    for col in X.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        logger.info("Encoded categorical column: %s", col)

    # Convert dates into numeric features
    if "GAME_DATE" in X.columns:
        X["GAME_DATE"] = pd.to_datetime(X["GAME_DATE"], errors="coerce")
        X["GAME_YEAR"] = X["GAME_DATE"].dt.year
        X["GAME_MONTH"] = X["GAME_DATE"].dt.month
        X["GAME_DAY"] = X["GAME_DATE"].dt.day
        X = X.drop(columns=["GAME_DATE"])
        logger.info("Extracted numeric features from GAME_DATE.")

    logger.info("Prepared data with X shape %s, y shape %s", X.shape, y.shape)
    return X, y


def train_model(model_type: str, X: pd.DataFrame, y: pd.Series, out_file: str, logger):
    """Train and save model, log evaluation metrics."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if model_type == "xgb":
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    elif model_type == "logreg":
        model = LogisticRegression(max_iter=1000)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds, target_names=["Loss", "Win"])

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    joblib.dump(model, out_file)

    logger.info("Model trained: %s", model_type)
    logger.info("Test accuracy: %.3f", acc)
    logger.info("Confusion matrix:\n%s", cm)
    logger.info("Classification report:\n%s", report)
    logger.info("Model saved to %s", out_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, required=True, help="Model type: xgb or logreg"
    )
    parser.add_argument("--season", type=int, required=True, help="Season year")
    parser.add_argument(
        "--features", type=str, required=True, help="Path to features parquet"
    )
    parser.add_argument("--out", type=str, required=True, help="Output model file path")
    args = parser.parse_args()

    logger = configure_logging(name="model_training.trainer_cli")
    logger.info("Starting training for model=%s, season=%s", args.model, args.season)

    df = pd.read_parquet(args.features)
    logger.info("Loaded features with shape %s", df.shape)

    try:
        X, y = preprocess_features(df, logger)
        train_model(args.model, X, y, args.out, logger)
    except Exception as e:
        logger.error("Error training model: %s", e)


if __name__ == "__main__":
    main()
