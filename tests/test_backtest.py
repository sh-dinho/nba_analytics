# Path: tests/test_backtest.py
import os
import pandas as pd
import pytest

import scripts.backtest as backtest
from nba_analytics_core.data import fetch_historical_games

RESULTS_DIR = "results"
BACKTEST_FILE = os.path.join(RESULTS_DIR, "backtest.csv")

@pytest.mark.parametrize("seasons", [["2021-22"]])
def test_backtest_creates_csv_and_matches_games(seasons):
    # Run backtest on a single season (smoke test)
    backtest.run_backtest(seasons)

    # Check that backtest file exists
    assert os.path.exists(BACKTEST_FILE), "Backtest CSV not created"

    # Load CSV and validate structure
    df = pd.read_csv(BACKTEST_FILE)
    assert not df.empty, "Backtest CSV is empty"
    assert "home_win_prob" in df.columns, "Missing probability column"
    assert df["home_win_prob"].between(0, 1).all(), "Probabilities not in [0,1]"

    # Fetch historical games for the same season
    games = fetch_historical_games(seasons)
    expected_rows = len(games)

    # Verify row count matches number of games
    assert len(df) == expected_rows, (
        f"Row count mismatch: expected {expected_rows}, got {len(df)}"
    )