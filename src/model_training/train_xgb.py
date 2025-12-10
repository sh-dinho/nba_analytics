# ============================================================
# File: src/model_training/train_xgb.py
# Purpose: Train XGBoost classifier for NBA prediction
# Project: nba_analysis
# Version: 1.3
# ============================================================

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
import joblib, os, logging
import pandas as pd

def train_xgb(cache_file, out_dir="models"):
    df = pd.read_parquet(cache_file)
    if "target" not in df.columns:
        logging.error("Target column missing")
        return None

    X = df.drop(columns=["target"], errors="ignore")
    y = df["target"]

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="mean"), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
        ]
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss",
            n_jobs=4
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)

    os.makedirs(out_dir, exist_ok=True)
    model_path = f"{out_dir}/nba_xgb.pkl"
    joblib.dump(pipeline, model_path)

    acc = accuracy_score(y_test, pipeline.predict(X_test))
    logging.info(f"XGB accuracy: {acc:.3f}")
    return {"model_path": model_path, "accuracy": acc}
