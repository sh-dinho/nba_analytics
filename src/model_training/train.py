# ============================================================
# File: src/model_training/train.py
# Purpose: Unified training for Logistic Regression and XGBoost
# Project: nba_analysis
# Version: 1.5
# ============================================================

import os
import joblib
import logging
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

def train_model(cache_file, model_type="logreg", out_dir="models"):
    df = pd.read_parquet(cache_file)
    if "win" not in df.columns:
        logging.error("Target column 'win' missing")
        return None

    X = df.drop(columns=["win"])
    y = df["win"]

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="mean"), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
        ]
    )

    if model_type == "logreg":
        clf = LogisticRegression(max_iter=1000)
        model_file = "logreg.pkl"
    else:
        pos = int(y.sum())
        neg = int(len(y) - pos)
        scale_pos_weight = (neg / pos) if pos > 0 else 1.0
        clf = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            n_jobs=4
        )
        model_file = "nba_xgb.pkl"

    pipeline = Pipeline([("preprocessor", preprocessor), ("clf", clf)])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(set(y)) > 1 else None
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_proba)
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, model_file)
    joblib.dump(pipeline, model_path)
    logging.info(f"{model_type} trained. Accuracy: {acc:.3f}, Logloss: {loss:.3f}")
    return model_path
