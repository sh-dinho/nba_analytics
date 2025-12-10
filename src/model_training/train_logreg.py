# ============================================================
# File: src/model_training/train_logreg.py
# Purpose: Train Logistic Regression with preprocessing
# ============================================================

import pandas as pd
import joblib, os, logging
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

def train_logreg(cache_file, out_dir="models"):
    df = pd.read_parquet(cache_file)
    if "target" not in df.columns:
        logging.error("Target column missing")
        return

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
        ("clf", LogisticRegression(max_iter=1000))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(set(y)) > 1 else None
    )

    pipeline.fit(X_train, y_train)

    acc = accuracy_score(y_test, pipeline.predict(X_test))
    loss = log_loss(y_test, pipeline.predict_proba(X_test))

    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "logreg.pkl")
    joblib.dump(pipeline, model_path)
    logging.info(f"Model saved to {model_path}, Accuracy: {acc:.3f}, LogLoss: {loss:.3f}")
