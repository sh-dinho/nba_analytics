# tests/test_engineer_features.py
import pandas as pd
from nba_analytics_core.data import engineer_features

def test_engineer_features_values():
    df = pd.DataFrame({
        "home_score": [100, 90],
        "away_score": [95, 85],
        "game_id": ["1", "2"]
    })
    res = engineer_features(df)
    assert "home_win" in res.columns
    assert "total_points" in res.columns
    assert list(res["home_win"]) == [True, True]
    assert list(res["total_points"]) == [195, 175]

def test_engineer_features_edge_cases():
    df = pd.DataFrame({
        "home_score": [100, None],
        "away_score": [100, 90],
        "game_id": ["3", "4"]
    })
    res = engineer_features(df)
    assert res.loc[res["game_id"] == "3", "home_win"].iloc[0] is False
    assert pd.isna(res.loc[res["game_id"] == "4", "home_win"]).iloc[0]