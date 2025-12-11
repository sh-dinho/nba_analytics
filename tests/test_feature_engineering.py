import pandas as pd
import pytest
from src.features.feature_engineering import generate_features_for_games


# Test for feature generation for games
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

    # Check that the generated DataFrame contains the expected number of rows
    assert features_df.shape[0] == 2  # Same number of rows as input

    # Check data types for numeric columns like RollingPTS_5 and TeamWinPctToDate
    assert pd.api.types.is_numeric_dtype(features_df["RollingPTS_5"])
    assert pd.api.types.is_numeric_dtype(features_df["TeamWinPctToDate"])

    # Check that the first row has expected values (based on the mock data, this should be manually verified)
    assert features_df["GAME_ID"].iloc[0] == "game_001"
    assert features_df["TEAM_ID"].iloc[0] == 1
    assert (
        features_df["RollingPTS_5"].iloc[0] >= 0
    )  # Assuming a positive number for rolling stats
    assert (
        features_df["TeamWinPctToDate"].iloc[0] >= 0
    )  # Assuming a valid win percentage


# Test for empty data
def test_generate_features_for_empty_data():
    empty_df = pd.DataFrame(
        columns=[
            "GAME_DATE",
            "TEAM_NAME",
            "MATCHUP",
            "POINTS",
            "TARGET",
            "TEAM_ID",
            "GAME_ID",
            "OPPONENT_TEAM_ID",
        ]
    )
    features_df = generate_features_for_games(empty_df.to_dict(orient="records"))
    assert features_df.empty  # Ensure it returns an empty DataFrame


# Test for single game data
def test_generate_features_for_single_game():
    single_game_data = [
        {
            "GAME_DATE": "2025-12-10",
            "TEAM_NAME": "Lakers",
            "MATCHUP": "Lakers vs. Celtics",
            "POINTS": 102,
            "TARGET": 1,
            "TEAM_ID": 1,
            "GAME_ID": "game_001",
            "OPPONENT_TEAM_ID": 2,
        }
    ]

    single_game_df = pd.DataFrame(single_game_data)
    features_df = generate_features_for_games(single_game_df.to_dict(orient="records"))

    # Only 1 game, so only 1 row
    assert features_df.shape[0] == 1

    # Check if rolling points are calculated (it should equal the game's points since it's a single game)
    assert "RollingPTS_5" in features_df.columns
    assert (
        pd.isna(features_df["RollingPTS_5"].iloc[0])
        or features_df["RollingPTS_5"].iloc[0] == 102
    )  # For single game, either NaN or the points themselves
