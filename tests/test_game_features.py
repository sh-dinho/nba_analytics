# ============================================================
# Path: tests/test_game_features.py
# Filename: test_game_features.py
# Author: Your Team
# Date: December 2025
# Purpose: Tests for game_features functions using real NBA data
# ============================================================

import pandas as pd
from features.game_features import (
    fetch_season_games,
    fetch_game_features,
    generate_features_for_games,
    _cache_path,
)

EXPECTED_COLUMNS = ["PTS", "REB", "AST", "FG_PCT", "FT_PCT", "PLUS_MINUS", "TOV", "win"]

def test_fetch_season_games_returns_ids():
    game_ids = fetch_season_games(2023, limit=3)
    assert isinstance(game_ids, list)
    assert len(game_ids) <= 3
    assert all(isinstance(gid, str) for gid in game_ids)

def test_fetch_game_features_structure():
    game_ids = fetch_season_games(2023, limit=1)
    df = fetch_game_features(game_ids[0])
    assert isinstance(df, pd.DataFrame)
    # Validate schema consistency
    for col in EXPECTED_COLUMNS:
        assert col in df.columns

def test_generate_features_for_games_combines_multiple():
    game_ids = fetch_season_games(2023, limit=2)
    df = generate_features_for_games(game_ids)
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 2  # At least one row per team per game
    # Validate schema consistency
    for col in EXPECTED_COLUMNS:
        assert col in df.columns

def test_cache_path_exists():
    assert _cache_path.exists()
    assert _cache_path.is_dir()
