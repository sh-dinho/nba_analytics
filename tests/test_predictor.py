import pandas as pd
from app.predictor import predict_todays_games


def test_predictor_runs():
    df = predict_todays_games()
    assert isinstance(df, pd.DataFrame)