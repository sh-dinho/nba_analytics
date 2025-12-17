# ============================================================
# File: tests/test_enrichment.py
# Purpose: Unit tests for feature engineering and enrichment
# Version: 2.0
# Author: Your Team
# Date: December 2025
# ============================================================

import pandas as pd
from src.schedule.pipeline_historical import (
    add_features,
    add_team_strength,
    add_predicted_win,
)


def test_enrichment_adds_features():
    df = pd.DataFrame(
        {
            "gameId": [1],
            "seasonYear": [2025],
            "startDate": ["2025-12-01"],
            "homeTeam": ["Lakers"],
            "awayTeam": ["Celtics"],
            "homeScore": [102],
            "awayScore": [99],
        }
    )
    df = add_features(df)
    df = add_team_strength(df)
    df = add_predicted_win(df)
    assert "predicted_win" in df.columns
    assert df["predicted_win"].notnull().all()
