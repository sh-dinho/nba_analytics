from src.model.predict_totals import _predict_totals
import pandas as pd


def test_totals_prediction():
    df = pd.DataFrame(
        {
            "game_id": ["G1"],
            "date": ["2024-01-01"],
            "home_team": ["A"],
            "away_team": ["B"],
            "f1": [1],
            "f2": [2],
        }
    )

    class DummyModel:
        def predict(self, X):
            return [220.5]

    out = _predict_totals(DummyModel(), df, ["f1", "f2"])
    assert out["predicted_total"].iloc[0] == 220.5
