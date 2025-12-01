# tests/test_config.py
import os
import json
import pytest
import importlib
import config

def test_db_path_env_override(monkeypatch):
    monkeypatch.setenv("DB_PATH", "custom/path.db")
    importlib.reload(config)
    assert config.DB_PATH == "custom/path.db"

def test_odds_api_key_env_override(monkeypatch):
    monkeypatch.setenv("ODDS_API_KEY", "SECRET_KEY")
    importlib.reload(config)
    assert config.ODDS_API_KEY == "SECRET_KEY"

def test_load_team_aliases_valid_file(tmp_path, monkeypatch):
    alias_file = tmp_path / "aliases.json"
    alias_file.write_text(json.dumps({"LAL": "Los Angeles Lakers"}))
    monkeypatch.setattr(config, "TEAM_ALIAS_PATH", str(alias_file))
    result = config.load_team_aliases()
    assert result["LAL"] == "Los Angeles Lakers"

def test_load_team_aliases_missing_file(monkeypatch, caplog):
    monkeypatch.setattr(config, "TEAM_ALIAS_PATH", "nonexistent.json")
    result = config.load_team_aliases()
    assert result == {}
    assert "Team alias file not found" in caplog.text

def test_load_team_aliases_invalid_json(tmp_path, monkeypatch, caplog):
    alias_file = tmp_path / "aliases.json"
    alias_file.write_text("not valid json")
    monkeypatch.setattr(config, "TEAM_ALIAS_PATH", str(alias_file))
    result = config.load_team_aliases()
    assert result == {}
    assert "Invalid team alias file" in caplog.text