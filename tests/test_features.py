# ============================================================
# Path: tests/test_features.py
# Purpose: Unit tests for feature generation
# Version: 1.1
# ============================================================

import pandas as pd
from src.features.generate_features import generate_features_for_games

def test_generate_features_with_ids():
    raw_data = [{"game_id": "123", "team_id": "456", "points": 100}]
    df = generate_features_for_games(raw_data)
    assert "GAME_ID" in df.columns
    assert "TEAM_ID" in df.columns
    assert df.loc[0, "GAME_ID"] == "123"
    assert df.loc[0, "TEAM_ID"] == "456"

def test_generate_features_missing_ids():
    raw_data = [{"points": 90}]
    df = generate_features_for_games(raw_data)
    assert "GAME_ID" in df.columns
    assert "TEAM_ID" in df.columns
    # Placeholder values should exist
    assert df.loc[0, "GAME_ID"].startswith("unknown_")
    assert df.loc[0, "TEAM_ID"] == -1
