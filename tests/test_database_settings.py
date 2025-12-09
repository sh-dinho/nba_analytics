# ============================================================
# Path: tests/test_database_settings.py
# Filename: test_database_settings.py
# Author: Your Team
# Date: December 2025
# Purpose: Tests for DatabaseSettings with and without env vars
# ============================================================

import pytest
from src.config_loader import load_settings, Settings

def test_database_settings_without_env(set_env):
    """
    When only NBA_API_KEY is set, database fields should be None.
    """
    settings = load_settings()
    assert isinstance(settings, Settings)
    assert settings.NBA_API_KEY == "dummy_key"
    assert settings.database.host is None
    assert settings.database.port is None
    assert settings.database.user is None
    assert settings.database.password is None

@pytest.mark.with_db
def test_database_settings_with_env(set_env):
    """
    When NBA_API_KEY and dummy database env vars are set,
    database fields should be populated.
    """
    settings = load_settings()
    assert isinstance(settings, Settings)
    assert settings.NBA_API_KEY == "dummy_key"
    assert settings.database.host == "localhost"
    assert settings.database.port == 5432
    assert settings.database.user == "test_user"
    assert settings.database.password == "test_pass"
