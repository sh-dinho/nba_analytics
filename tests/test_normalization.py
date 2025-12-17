# ============================================================
# File: tests/test_normalization.py
# Purpose: Unit tests for schema normalization
# Version: 2.0
# Author: Your Team
# Date: December 2025
# ============================================================

import pandas as pd
from src.schemas.normalize import normalize


def test_normalize_enriched_schedule():
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
    normalized = normalize(df, "enriched_schedule")
    assert "gameId" in normalized.columns
    assert "homeTeam" in normalized.columns
