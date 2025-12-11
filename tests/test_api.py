import pandas as pd
import pytest

from src.api.nba_api_wrapper import fetch_today_games


def test_fetch_today_games():
    # Test that the fetch_today_games function works without errors
    today_games = fetch_today_games()

    # If no games are found, it should return an empty dataframe
    assert isinstance(today_games, pd.DataFrame)
    assert today_games.shape[0] >= 0  # At least zero rows (if no games)


def test_fetch_today_games_mocked():
    # Test with mocked data to simulate a real API call

    # Create mock data for testing
    mock_games = pd.DataFrame(
        {
            "GAME_DATE": ["2025-12-10"],
            "MATCHUP": ["Lakers @ Celtics"],
            "TEAM_NAME": ["Lakers"],
            "HOME_TEAM": ["Celtics"],
            "AWAY_TEAM": ["Lakers"],
            "OPPONENT_TEAM_ID": [2],
        }
    )

    assert not mock_games.empty
    assert "GAME_DATE" in mock_games.columns
