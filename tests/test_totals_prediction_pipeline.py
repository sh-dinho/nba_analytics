import pandas as pd
from datetime import date
from unittest.mock import MagicMock

from src.model.predict_totals import run_totals_prediction_for_date


def test_totals_prediction_pipeline(monkeypatch):
    monkeypatch.setattr(
        "src.model.predict_totals.load_model_and_metadata",
        lambda t="totals": (
            MagicMock(predict=lambda X: [218.5]),
            {
                "model_name": "tot_test",
                "version": "v1",
                "feature_version": "v1",
                "feature_cols": ["f1", "f2"],
            },
        ),
    )

    monkeypatch.setattr(
        "src.model.predict_totals._build_prediction_features",
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

    df = run_totals_prediction_for_date(date(2024, 1, 1))
    assert not df.empty
    assert df["predicted_total"].iloc[0] == 218.5
