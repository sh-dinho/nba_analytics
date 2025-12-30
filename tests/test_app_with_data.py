import pandas as pd
from datetime import date
from streamlit.testing.v1 import AppTest


def test_app_with_combined_predictions(tmp_path, monkeypatch):
    # Redirect combined predictions dir
    monkeypatch.setattr("src.dashboard.app.COMBINED_PRED_DIR", tmp_path)

    pred_date = date(2025, 1, 1)

    # Override today's date
    monkeypatch.setattr("src.dashboard.app.get_today", lambda: pred_date)

    # Create fake combined predictions
    df = pd.DataFrame(
        {
            "game_id": ["g1"],
            "home_team": ["A"],
            "away_team": ["B"],
            "win_probability_home": [0.65],
            "win_probability_away": [0.35],
            "predicted_total_points": [228.5],
            "predicted_margin": [4.5],
            "model_version": ["vtest"],
        }
    )

    path = tmp_path / f"predictions_combined_{pred_date}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)

    at = AppTest.from_file("src/dashboard/app.py")
    at.run()

    assert not at.exception

    rendered = " ".join(el.value for el in at.text)
    assert "A" in rendered
    assert "B" in rendered
