import os
import logging
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from config import TRAINING_FEATURES_FILE, MODEL_FILE

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

def main():
    if not os.path.exists(TRAINING_FEATURES_FILE):
        raise FileNotFoundError(f"{TRAINING_FEATURES_FILE} not found. Run build_features.py first.")

    df = pd.read_csv(TRAINING_FEATURES_FILE)
    if "home_win" not in df.columns:
        raise ValueError("Training data missing 'home_win' column.")

    logger.info("Training model...")

    X = df.drop(columns=["home_win"])
    y = df["home_win"].fillna(0).astype(int)

    X_num = X.select_dtypes(include="number")
    X_num = X_num.fillna(0).replace([float("inf"), -float("inf")], 0)

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_num, y)

    # Save both model and feature order
    feature_order = list(X_num.columns)
    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    joblib.dump({"model": model, "features": feature_order}, MODEL_FILE)

    logger.info(f"✅ Model trained and saved to {MODEL_FILE} with feature order")

if __name__ == "__main__":
    main()