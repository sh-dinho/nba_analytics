import pandas as pd
from datetime import date

from src.dashboard.app import (
    _load_moneyline,
    _load_totals,
    _load_spread,
    _load_combined,
    _combine_predictions,
)
from src.config.paths import (
    PREDICTIONS_DIR,
    TOTALS_PRED_DIR,
    SPREAD_PRED_DIR,
    COMBINED_PRED_DIR,
)


def test_load_moneyline_reads_file(tmp_path, monkeypatch):
    # Redirect predictions dir to tmp
    monkeypatch.setattr("src.dashboard.app.PREDICTIONS_DIR", tmp_path)

    pred_date = date(2025, 1, 1)
    df = pd.DataFrame(
        {
            "game_id": ["g1", "g1"],
            "team": ["A", "B"],
            "opponent": ["B", "A"],
            "is_home": [1, 0],
            "win_probability": [0.6, 0.4],
        }
    )

    path = tmp_path / f"predictions_moneyline_{pred_date}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)

    loaded = _load_moneyline(pred_date)
    assert not loaded.empty
    assert set(loaded.columns) == set(df.columns)


def test_load_moneyline_missing_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr("src.dashboard.app.PREDICTIONS_DIR", tmp_path)

    pred_date = date(2025, 1, 1)
    loaded = _load_moneyline(pred_date)
    assert loaded.empty


def test_combine_predictions_fallback(tmp_path, monkeypatch):
    # Redirect all prediction dirs to tmp
    monkeypatch.setattr("src.dashboard.app.PREDICTIONS_DIR", tmp_path)
    monkeypatch.setattr("src.dashboard.app.TOTALS_PRED_DIR", tmp_path)
    monkeypatch.setattr("src.dashboard.app.SPREAD_PRED_DIR", tmp_path)
    monkeypatch.setattr("src.dashboard.app.COMBINED_PRED_DIR", tmp_path)

    pred_date = date(2025, 1, 1)

    # Moneyline long-format
    ml = pd.DataFrame(
        {
            "game_id": ["g1", "g1"],
            "date": [pred_date, pred_date],
            "team": ["A", "B"],
            "opponent": ["B", "A"],
            "is_home": [1, 0],
            "win_probability": [0.7, 0.3],
        }
    )
    (tmp_path / f"predictions_moneyline_{pred_date}.parquet").parent.mkdir(
        parents=True, exist_ok=True
    )
    ml.to_parquet(tmp_path / f"predictions_moneyline_{pred_date}.parquet")

    # Totals
    totals = pd.DataFrame(
        {
            "game_id": ["g1"],
            "home_team": ["A"],
            "away_team": ["B"],
            "predicted_total_points": [225.5],
        }
    )
    totals.to_parquet(tmp_path / f"totals_{pred_date}.parquet")

    # Spread
    spread = pd.DataFrame(
        {
            "game_id": ["g1"],
            "home_team": ["A"],
            "away_team": ["B"],
            "predicted_margin": [5.5],
        }
    )
    spread.to_parquet(tmp_path / f"spread_{pred_date}.parquet")

    combined = _combine_predictions(pred_date)
    assert not combined.empty
    assert "win_probability_home" in combined.columns
    assert "win_probability_away" in combined.columns
    assert "predicted_total_points" in combined.columns
    assert "predicted_margin" in combined.columns
