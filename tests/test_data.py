import pandas as pd
from nba_analytics_core.data import engineer_features


def test_engineer_features():
    df = pd.DataFrame({
        "home_score": [100, 90],
        "away_score": [95, 85]
    })
    df["game_id"] = ["1", "2"]
    result = engineer_features(df)
    assert "home_win" in result.columns
    assert "total_points" in result.columns