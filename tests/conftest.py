import pandas as pd
import pytest


@pytest.fixture
def sample_game_data():
    # Fixture to provide sample game data
    return pd.DataFrame(
        {
            "GAME_DATE": ["2025-12-10", "2025-12-09"],
            "TEAM_NAME": ["Lakers", "Celtics"],
            "MATCHUP": ["Lakers vs. Celtics", "Celtics vs. Lakers"],
            "POINTS": [102, 99],
            "TARGET": [1, 0],
            "TEAM_ID": [1, 2],
            "GAME_ID": ["game_001", "game_002"],
            "OPPONENT_TEAM_ID": [2, 1],
        }
    )


def test_sample_game_data_columns(sample_game_data):
    # Test if the correct columns are present
    assert "GAME_DATE" in sample_game_data.columns
    assert "TEAM_NAME" in sample_game_data.columns
    assert "MATCHUP" in sample_game_data.columns
    assert "POINTS" in sample_game_data.columns
    assert "TARGET" in sample_game_data.columns
    assert "TEAM_ID" in sample_game_data.columns
    assert "GAME_ID" in sample_game_data.columns
    assert "OPPONENT_TEAM_ID" in sample_game_data.columns


def test_sample_game_data_row_count(sample_game_data):
    # Test if the data contains the correct number of rows
    assert len(sample_game_data) == 2  # There should be 2 rows in the sample data


def test_target_column_values(sample_game_data):
    # Test if the 'TARGET' column contains expected values (e.g., binary outcome)
    assert set(sample_game_data["TARGET"]) == {0, 1}  # Ensure TARGET is binary


def test_calculate_team_win_percentage(sample_game_data):
    # Example of a calculation you might perform: calculate win percentage
    sample_game_data["TEAM_WIN_PERCENTAGE"] = (
        sample_game_data.groupby("TEAM_NAME")["POINTS"]
        .expanding()
        .mean()
        .reset_index(level=0, drop=True)
    )

    # Test if the calculated win percentage column exists
    assert "TEAM_WIN_PERCENTAGE" in sample_game_data.columns

    # Test if the win percentages are correct (e.g., it should be the same as POINTS for this test)
    assert sample_game_data.loc[0, "TEAM_WIN_PERCENTAGE"] == 102
    assert sample_game_data.loc[1, "TEAM_WIN_PERCENTAGE"] == 99


def test_matchup_format(sample_game_data):
    # Test if the 'MATCHUP' column is in the expected format
    assert sample_game_data["MATCHUP"].iloc[0] == "Lakers vs. Celtics"
    assert sample_game_data["MATCHUP"].iloc[1] == "Celtics vs. Lakers"


def test_game_date_format(sample_game_data):
    # Test if the 'GAME_DATE' column is in the correct datetime format
    sample_game_data["GAME_DATE"] = pd.to_datetime(sample_game_data["GAME_DATE"])
    assert (
        sample_game_data["GAME_DATE"].dtype == "datetime64[ns]"
    )  # Ensure it's datetime type


if __name__ == "__main__":
    pytest.main()
