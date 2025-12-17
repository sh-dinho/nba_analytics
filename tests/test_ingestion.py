# ============================================================
# File: tests/test_pipeline_ingestion_schema.py
# Purpose: Test ingestion alignment from raw NBA boxscore schema to canonical schema
# Version: 2.0
# Author: Your Team
# Date: December 2025
# ============================================================

import pandas as pd
from pathlib import Path
from src.pipeline_runner import align_nba_boxscore_schema


def test_align_nba_boxscore_schema(tmp_path):
    # Create sample raw NBA boxscore data
    df_raw = pd.DataFrame(
        {
            "SEASON_ID": [2025, 2025],
            "TEAM_ID": [1610612747, 1610612738],
            "TEAM_ABBREVIATION": ["LAL", "BOS"],
            "TEAM_NAME": ["Lakers", "Celtics"],
            "GAME_ID": ["002250001", "002250001"],
            "GAME_DATE": ["2025-12-01", "2025-12-01"],
            "MATCHUP": ["LAL vs BOS", "BOS @ LAL"],
            "WL": ["W", "L"],
            "PTS": [102, 99],
        }
    )

    # Align schema
    df_aligned = align_nba_boxscore_schema(df_raw)

    # Assertions
    assert not df_aligned.empty
    assert {
        "gameId",
        "seasonYear",
        "startDate",
        "homeTeam",
        "awayTeam",
        "homeScore",
        "awayScore",
    } <= set(df_aligned.columns)

    # Check values
    row = df_aligned.iloc[0]
    assert row["homeTeam"] == "LAL"
    assert row["awayTeam"] == "BOS"
    assert row["homeScore"] == 102
    assert row["awayScore"] == 99
