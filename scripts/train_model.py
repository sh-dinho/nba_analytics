# File: scripts/train_model.py
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
import joblib
import os

def main():
    # Load your training data
    df = pd.read_csv("data/training_data.csv")  # adjust path

    X = df.drop(columns=["target"])
    y = df["target"]

    # Build pipeline: imputer + logistic regression
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X, y)

    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, "models/game_predictor.pkl")
    print("✅ Model trained and saved with imputer included.")

if __name__ == "__main__":
    main()