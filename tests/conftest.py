# ============================================================
# Path: tests/conftest.py
# Filename: conftest.py
# Author: Your Team
# Date: December 2025
# Purpose: Shared pytest fixtures for NBA analytics tests
# ============================================================

import pytest
import pandas as pd
from src.prediction_engine.game_features import fetch_season_games, fetch_game_features

@pytest.fixture(scope="function")
def real_features(tmp_path):
    """
    Fetch a small batch of real NBA games from the 2023 season
    and save them as a parquet file for testing.
    """
    game_ids = fetch_season_games(2023, limit=3)
    features = pd.concat([fetch_game_features(gid) for gid in game_ids], ignore_index=True)

    features_path = tmp_path / "features.parquet"
    features.to_parquet(features_path, index=False)
    return str(features_path)

@pytest.fixture
def model_dir(tmp_path):
    """Provide a temporary directory for saving models."""
    out_dir = tmp_path / "models"
    out_dir.mkdir()
    return str(out_dir)

@pytest.fixture
def set_env(monkeypatch, request):
    """
    Fixture to set environment variables.
    By default sets NBA_API_KEY only.
    If test is marked with 'with_db', also sets dummy database values
    using both prefixed (DATABASE_HOST) and short (HOST) names.
    """
    monkeypatch.setenv("NBA_API_KEY", "dummy_key")

    if request.node.get_closest_marker("with_db"):
        # Prefixed env vars
        monkeypatch.setenv("DATABASE_HOST", "localhost")
        monkeypatch.setenv("DATABASE_PORT", "5432")
        monkeypatch.setenv("DATABASE_USER", "test_user")
        monkeypatch.setenv("DATABASE_PASSWORD", "test_pass")

        # Short env vars (optional fallback)
        monkeypatch.setenv("HOST", "localhost")
        monkeypatch.setenv("PORT", "5432")
        monkeypatch.setenv("USER", "test_user")
        monkeypatch.setenv("PASSWORD", "test_pass")

    yield

    # Clean up
    for var in [
        "NBA_API_KEY",
        "DATABASE_HOST", "DATABASE_PORT", "DATABASE_USER", "DATABASE_PASSWORD",
        "HOST", "PORT", "USER", "PASSWORD"
    ]:
        monkeypatch.delenv(var, raising=False)
