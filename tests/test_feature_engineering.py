# ============================================================
# File: tests/test_feature_engineering.py
# Purpose: Unit tests for feature_engineering.generate_features_for_games
# Project: nba_analysis
# ============================================================

import pandas as pd
import numpy as np
import pytest

from src.features.feature_engineering import (
    generate_features_for_games,
    EXPECTED_COLUMNS,
)


def test_empty_input_returns_expected_schema():
    df = generate_features_for_games([])
    assert list(df.columns) == EXPECTED_COLUMNS
    assert df.empty


def test_dict_input_basic_fields():
    games = [{"GAME_ID": "001", "TEAM_ID": 1610612747}]
    df = generate_features_for_games(games)
    assert "unique_id" in df.columns
    assert df.iloc[0]["GAME_ID"] == "001"
    assert df.iloc[0]["TEAM_ID"] == 1610612747


def test_dataframe_input_with_game_date_and_points():
    df_in = pd.DataFrame(
        {
            "GAME_ID": ["001", "002"],
            "TEAM_ID": [1, 1],
            "GAME_DATE": ["2025-12-01", "2025-12-05"],
            "POINTS": [100, 110],
            "TARGET": [1, 0],
        }
    )
    df_out = generate_features_for_games(df_in)
    assert "RollingPTS_5" in df_out.columns
    assert not df_out["RollingPTS_5"].isna().all()
    assert "RollingWinPct_10" in df_out.columns
    assert not df_out["RollingWinPct_10"].isna().all()
    assert "RestDays" in df_out.columns
    assert df_out["RestDays"].iloc[1] == 4  # 5th - 1st December


def test_player_points_dict_and_list():
    games = [
        {"GAME_ID": "003", "TEAM_ID": 2, "PLAYER_POINTS": {"p1": 25, "p2": 10}},
        {"GAME_ID": "004", "TEAM_ID": 3, "PLAYER_POINTS": [12, 22, 30]},
        {
            "GAME_ID": "005",
            "TEAM_ID": 4,
            "PLAYER_POINTS": [{"pid": 1, "pts": 28}, {"pid": 2, "pts": 15}],
        },
    ]
    df = generate_features_for_games(games)
    assert all(df["Players20PlusPts"] >= 1)


def test_opponent_win_pct_merge():
    df_in = pd.DataFrame(
        {
            "GAME_ID": ["006", "007"],
            "TEAM_ID": [1, 2],
            "OPPONENT_TEAM_ID": [2, 1],
            "GAME_DATE": ["2025-12-01", "2025-12-02"],
            "TARGET": [1, 0],
        }
    )
    df_out = generate_features_for_games(df_in)
    assert "OppWinPctToDate" in df_out.columns
    # Opponent win pct should be numeric or NaN
    assert pd.api.types.is_numeric_dtype(df_out["OppWinPctToDate"])


def test_expected_columns_always_present():
    games = [{"GAME_ID": "008", "TEAM_ID": 5}]
    df = generate_features_for_games(games)
    for col in EXPECTED_COLUMNS:
        assert col in df.columns
