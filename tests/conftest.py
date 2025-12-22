import pandas as pd
import pytest
from datetime import date


@pytest.fixture
def sample_long_df():
    return pd.DataFrame(
        {
            "game_id": ["G1", "G1", "G2", "G2"],
            "team": ["A", "B", "C", "D"],
            "date": [date(2024, 1, 1)] * 4,
            "points_for": [100, 98, 110, 105],
            "points_against": [98, 100, 105, 110],
            "is_home": [True, False, True, False],
            "won": [1, 0, 1, 0],
        }
    )
