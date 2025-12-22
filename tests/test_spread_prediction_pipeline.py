import pandas as pd
from datetime import date
from unittest.mock import MagicMock

from src.model.predict_spread import run_spread_prediction_for_date


def test_spread_prediction_pipeline(monkeypatch):
    monkeypatch.setattr(
        "src.model.predict_spread.load_model_and_metadata",
        lambda t="spread": (
            MagicMock(predict=lambda X: [4.5]),
            {
                "model_name": "sp_test",
                "version": "v1",
                "feature_version": "v1",
                "feature_cols": ["f1", "f2"],
            },
        ),
    )

    monkeypatch.setattr(
        "src.model.predict_spread._build_prediction_features",
        lambda d, fv: pd.DataFrame(
            {
                "game_id": ["G1"],
                "date": [d],
                "home_team": ["A"],
                "away_team": ["B"],
                "f1": [1],
                "f2": [2],
            }
        ),
    )

    df = run_spread_prediction_for_date(date(2024, 1, 1))
    assert not df.empty
    assert df["predicted_margin"].iloc[0] == 4.5
