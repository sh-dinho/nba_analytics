import logging
import pandas as pd
import joblib

from core.data import engineer_features
from core.fetch_games import get_todays_games


def load_models():
    clf = joblib.load("models/classification_model.pkl")
    reg = joblib.load("models/regression_model.pkl")
    return clf, reg


def predict_todays_games():
    games = get_todays_games()
    if games.empty:
        logging.info("No games found for today.")
        return pd.DataFrame()

    df_features = engineer_features(games)
    clf, reg = load_models()

    df_features["pred_home_win_prob"] = clf.predict_proba(
        df_features.drop(["home_win", "total_points"], axis=1)
    )[:, 1]
    df_features["pred_total_points"] = reg.predict(
        df_features.drop(["home_win", "total_points"], axis=1)
    )

    df_pred = games.merge(
        df_features[["game_id", "pred_home_win_prob", "pred_total_points"]],
        on="game_id"
    )
    return df_pred