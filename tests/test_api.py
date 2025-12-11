import pytest
from unittest.mock import patch
from datetime import datetime
import pandas as pd
from src.api.nba_api_client import fetch_games


# Test for the fetch_games function
@patch("src.api.nba_api_client.requests.get")
def test_fetch_games(mock_get):
    # Mock the API response
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {
        "lscd": [
            {
                "mscd": {
                    "g": [
                        {
                            "gid": "001",
                            "h": {"tid": 1},
                            "v": {"tid": 2},
                            "gdte": "2023-12-10",
                        }
                    ]
                }
            }
        ]
    }

    # Call the function
    date = "2023-12-10"
    games_df = fetch_games(date, use_cache=False)

    # Validate the output
    assert isinstance(games_df, pd.DataFrame)  # Check if it returns a DataFrame
    assert games_df.shape[0] > 0  # Ensure there are some rows
    assert "GAME_ID" in games_df.columns  # Ensure that GAME_ID column is present
    assert games_df["GAME_ID"].iloc[0] == "001"  # Ensure the GAME_ID is as expected
    assert games_df["date"].iloc[0] == date  # Ensure the date is as expected
