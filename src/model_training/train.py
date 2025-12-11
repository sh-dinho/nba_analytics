# ============================================================
# File: src/model_training/train.py
# Purpose: Train NBA prediction models on engineered features
# Project: nba_analysis
# Version: 2.0 (schedule-based features, safe preprocessing)
# ============================================================

import logging
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import xgboost as xgb

logger = logging.getLogger("model_training.train")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def _prepare_data(features_df: pd.DataFrame):
    """
    Prepare X and y for training.
    Expects schedule-based features with HOME_GAME, DAYS_SINCE_GAME, and optional win target.
    """
    df = features_df.copy()

    # Target variable
    if "win" not in df.columns:
        logger.warning("No 'win' column found. Using HOME_GAME as proxy target.")
        df["win"] = df["HOME_GAME"]

    y = df["win"]

    # Feature set: numeric/categorical
    feature_cols = []
    for col in ["HOME_GAME", "DAYS_SINCE_GAME"]:
        if col in df.columns:
            feature_cols.append(col)

    # Encode TEAM_NAME as categorical if present
    if "TEAM_NAME" in df.columns:
        df["TEAM_NAME"] = df["TEAM_NAME"].astype("category").cat.codes
        feature_cols.append("TEAM_NAME")

    X = df[feature_cols]

    logger.info("Prepared data with X shape %s, y shape %s", X.shape, y.shape)
    return X, y


def train_model(model_type: str, features_df: pd.DataFrame):
    """
    Train a model (logreg or xgb) on the provided features DataFrame.
    """
    X, y = _prepare_data(features_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if model_type == "logreg":
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        logger.info("Logistic Regression accuracy: %.3f", acc)
        return model

    elif model_type == "xgb":
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        logger.info("XGBoost accuracy: %.3f", acc)
        return model

    else:
        raise ValueError(f"Unsupported model type: {model_type}")
