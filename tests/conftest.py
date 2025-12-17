# ============================================================
# File: tests/conftest.py
# Purpose: Pytest fixtures for pipeline e2e (v2.0)
# Version: 2.0
# Author: Your Team
# Date: December 2025
# ============================================================

import pytest
from pathlib import Path
import pandas as pd


@pytest.fixture(scope="session", autouse=True)
def prepare_history(tmp_path_factory):
    Path("data/history").mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "gameId": [1, 2, 3],
            "seasonYear": [2025, 2025, 2025],
            "startDate": ["2025-12-01", "2025-12-02", "2025-12-03"],
            "homeTeam": ["Lakers", "Warriors", "Raptors"],
            "awayTeam": ["Celtics", "Suns", "Knicks"],
            "homeScore": [102, 110, 98],
            "awayScore": [99, 108, 95],
        }
    )
    df.to_parquet("data/history/historical_schedule.parquet", index=False)
    yield
