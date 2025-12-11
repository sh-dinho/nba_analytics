import pandas as pd
import pytest


@pytest.fixture
def sample_game_data():
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
