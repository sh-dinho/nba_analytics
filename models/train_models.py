import logging
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression

from core.data import fetch_historical_games, engineer_features


def train_models_cached():
    df = fetch_historical_games()
    df = engineer_features(df)

    X = df.drop(["home_win", "total_points"], axis=1)
    y_class = df["home_win"]
    y_reg = df["total_points"]

    clf = LogisticRegression(max_iter=1000).fit(X, y_class)
    reg = LinearRegression().fit(X, y_reg)

    joblib.dump(clf, "models/classification_model.pkl")
    joblib.dump(reg, "models/regression_model.pkl")

    logging.info("✔ Models trained and saved")