# tests/test_predictor.py
import pandas as pd
from nba_analytics_core.db_module import init_db, insert_games
from nba_analytics_core.predictor import predict_todays_games

def test_predictor_runs_and_schema():
    init_db()
    insert_games([
        {"game_id": "X-1", "season": 2025, "date": "2025-01-01", "home_team": "LAL", "away_team": "BOS", "home_score": 100, "away_score": 98}
    ])
    df = predict_todays_games(threshold=0.6)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    expected_cols = {"game_id", "home_team", "away_team", "predicted_prob", "predicted_winner"}
    assert expected_cols.issubset(df.columns)
    assert df["predicted_prob"].between(0, 1).all()