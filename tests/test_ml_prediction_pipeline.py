import pandas as pd
from datetime import date
from unittest.mock import MagicMock

from src.model.predict import run_prediction_for_date


def test_ml_prediction_pipeline(monkeypatch, tmp_path):
    # Mock model registry
    monkeypatch.setattr(
        "src.model.predict.load_model_and_metadata",
        lambda: (
            MagicMock(predict=lambda X: [0.55] * len(X)),
            {
                "model_name": "ml_test",
                "version": "v1",
                "feature_version": "v1",
                "feature_cols": ["f1", "f2"],
            },
        ),
    )

    # Mock feature builder
    monkeypatch.setattr(
        "src.model.predict._build_prediction_features",
        lambda d, fv: pd.DataFrame(
            {
                "game_id": ["G1"],
                "team": ["A"],
                "opponent": ["B"],
                "date": [d],
                "f1": [1],
                "f2": [2],
            }
        ),
    )

    df = run_prediction_for_date(date(2024, 1, 1))
    assert not df.empty
    assert "win_probability" in df.columns
    assert df["win_probability"].iloc[0] == 0.55
