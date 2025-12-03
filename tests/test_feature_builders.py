# ============================================================
# File: tests/test_feature_builders.py
# Purpose: Unit tests for feature builder scripts
# ============================================================

import os
import pandas as pd
import pytest

from scripts.build_features_for_training import build_features_for_training
from scripts.build_features_for_new_games import build_features_for_new_games
from core.config import TRAINING_FEATURES_FILE, HISTORICAL_GAMES_FILE, NEW_GAMES_FILE


def test_build_features_for_training(tmp_path):
    # Create synthetic historical data
    hist_file = tmp_path / "historical_games.csv"
    df = pd.DataFrame([
        {"PLAYER_NAME": "LeBron James", "TEAM_ABBREVIATION": "LAL", "TEAM_HOME": "LAL", "TEAM_AWAY": "BOS",
         "PTS": 28, "AST": 8, "REB": 9, "GAMES_PLAYED": 20, "HOME_WIN": 1},
        {"PLAYER_NAME": "Jayson Tatum", "TEAM_ABBREVIATION": "BOS", "TEAM_HOME": "LAL", "TEAM_AWAY": "BOS",
         "PTS": 26, "AST": 5, "REB": 7, "GAMES_PLAYED": 19, "HOME_WIN": 1},
    ])
    df.to_csv(hist_file, index=False)

    # Run builder
    out_file = build_features_for_training(historical_file=str(hist_file))
    assert os.path.exists(out_file)

    features = pd.read_csv(out_file)
    expected_cols = {
        "game_id", "home_team", "away_team",
        "home_avg_pts", "home_avg_ast", "home_avg_reb", "home_avg_games_played",
        "away_avg_pts", "away_avg_ast", "away_avg_reb", "away_avg_games_played",
        "label"
    }
    assert expected_cols.issubset(features.columns)


def test_build_features_for_new_games(tmp_path):
    # Create synthetic new games data
    new_file = tmp_path / "new_games.csv"
    df = pd.DataFrame([
        {"PLAYER_NAME": "Stephen Curry", "TEAM_ABBREVIATION": "GSW", "TEAM_HOME": "GSW", "TEAM_AWAY": "MIA",
         "PTS": 30, "AST": 7, "REB": 5, "GAMES_PLAYED": 21, "decimal_odds": 1.9},
        {"PLAYER_NAME": "Jimmy Butler", "TEAM_ABBREVIATION": "MIA", "TEAM_HOME": "GSW", "TEAM_AWAY": "MIA",
         "PTS": 25, "AST": 6, "REB": 7, "GAMES_PLAYED": 19, "decimal_odds": 1.9},
    ])
    df.to_csv(new_file, index=False)

    # Run builder
    out_file = build_features_for_new_games(new_games_file=str(new_file))
    assert os.path.exists(out_file)

    features = pd.read_csv(out_file)
    expected_cols = {
        "game_id", "home_team", "away_team",
        "home_avg_pts", "home_avg_ast", "home_avg_reb", "home_avg_games_played",
        "away_avg_pts", "away_avg_ast", "away_avg_reb", "away_avg_games_played",
        "decimal_odds"
    }
    assert expected_cols.issubset(features.columns)