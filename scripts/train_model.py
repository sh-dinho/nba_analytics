# File: scripts/train_model.py

import os
import logging
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

def main():
    features_file = "data/training_features.csv"
    if not os.path.exists(features_file):
        raise FileNotFoundError(f"{features_file} not found. Run build_features.py first.")

    df = pd.read_csv(features_file)

    if "home_win" not in df.columns:
        raise ValueError("Training data missing 'home_win' column. "
                         "Ensure build_features.py merges game outcomes or adds synthetic labels.")

    logger.info("Training model...")
    X = df.drop(columns=["home_win"])
    y = df["home_win"].fillna(0).astype(int)

    # Example: simple model training placeholder
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter=1000)
    model.fit(X.select_dtypes(include="number"), y)

    logger.info("✅ Model trained successfully")

if __name__ == "__main__":
    main()