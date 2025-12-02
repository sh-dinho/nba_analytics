# ============================================================
# File: app/train_model.py
# Purpose: Train NBA game predictor model (logistic, XGB, NN) and save artifact
# ============================================================

import os
import argparse
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Optional imports for other models
try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
except ImportError:
    tf = None


# Ensure models directory exists
os.makedirs("models", exist_ok=True)


def load_data():
    """
    Load training data. Replace with your actual dataset.
    For now, we use a synthetic example.
    """
    data = pd.DataFrame({
        "feature1": [0.1, 0.2, 0.8, 0.9, 0.4, 0.6],
        "feature2": [1, 2, 3, 4, 5, 6],
        "label":    [0, 0, 1, 1, 0, 1]
    })
    return data


def train_logistic(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model


def train_xgb(X_train, y_train):
    if xgb is None:
        raise ImportError("XGBoost not installed in this environment")
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)
    return model


def train_nn(X_train, y_train, input_dim):
    if tf is None:
        raise ImportError("TensorFlow not installed in this environment")
    model = Sequential([
        Dense(16, activation="relu", input_dim=input_dim),
        Dense(8, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=20, batch_size=4, verbose=0)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="logistic",
                        choices=["logistic", "xgb", "nn"],
                        help="Type of model to train")
    args = parser.parse_args()

    # Load data
    data = load_data()
    X = data[["feature1", "feature2"]]
    y = data["label"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Train selected model
    if args.model_type == "logistic":
        model = train_logistic(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"✅ Logistic Regression Accuracy: {acc:.2f}")
        joblib.dump(model, "models/game_predictor.pkl")

    elif args.model_type == "xgb":
        model = train_xgb(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"✅ XGBoost Accuracy: {acc:.2f}")
        joblib.dump(model, "models/game_predictor.pkl")

    elif args.model_type == "nn":
        model = train_nn(X_train, y_train, input_dim=X_train.shape[1])
        _, acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"✅ Neural Network Accuracy: {acc:.2f}")
        # Save NN model in HDF5 format
        model.save("models/game_predictor.h5")
        print("📦 NN model saved to models/game_predictor.h5")

    print("📦 Training complete. Model artifact saved.")


if __name__ == "__main__":
    main()