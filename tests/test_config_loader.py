# ============================================================
# Path: tests/test_config_loader.py
# Purpose: Unit tests for config_loader.py with .env and JSON fallback
# Project: nba_analysis
# ============================================================

import pytest
from pydantic import ValidationError
from src.config_loader import load_settings, Settings

def test_load_settings_valid(set_env):
    settings = load_settings()
    assert isinstance(settings, Settings)
    assert settings.NBA_API_KEY == "dummy_key"

def test_load_settings_invalid(monkeypatch):
    monkeypatch.delenv("NBA_API_KEY", raising=False)
    with pytest.raises(ValidationError):
        load_settings()
