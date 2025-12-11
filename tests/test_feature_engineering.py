import pandas as pd
import pytest

from src.features.feature_engineering import generate_features_for_games


def test_generate_features_for_games():
    # Create mock data for testing
    game_data = [
        {
            "GAME_DATE": "2025-12-10",
            "TEAM_NAME": "Lakers",
            "MATCHUP": "Lakers vs. Celtics",
            "POINTS": 102,
            "TARGET": 1,
            "TEAM_ID": 1,
            "GAME_ID": "game_001",
            "OPPONENT_TEAM_ID": 2,
        },
        {
            "GAME_DATE": "2025-12-09",
            "TEAM_NAME": "Celtics",
            "MATCHUP": "Celtics vs. Lakers",
            "POINTS": 99,
            "TARGET": 0,
            "TEAM_ID": 2,
            "GAME_ID": "game_002",
            "OPPONENT_TEAM_ID": 1,
        },
    ]

    df_games = pd.DataFrame(game_data)

    # Generate features
    features_df = generate_features_for_games(df_games.to_dict(orient="records"))

    # Check if the generated DataFrame has the expected columns
    assert "GAME_ID" in features_df.columns
    assert "TEAM_ID" in features_df.columns
    assert "RollingPTS_5" in features_df.columns
    assert "TeamWinPctToDate" in features_df.columns

    # Check for expected shape
    assert features_df.shape[0] == 2  # Same number of rows as input
