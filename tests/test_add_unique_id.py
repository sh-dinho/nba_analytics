# ============================================================
# File: tests/test_add_unique_id.py
# Purpose: Validate unique_id generation and deduplication
# Project: nba_analysis
# ============================================================

import datetime
import pandas as pd
import pytest

from src.utils.add_unique_id import add_unique_id


def test_add_unique_id_basic():
    df = pd.DataFrame(
        {"GAME_ID": ["g1"], "TEAM_ID": [1], "prediction_date": ["2025-12-11"]}
    )
    out = add_unique_id(df)
    assert "unique_id" in out.columns
    assert out.loc[0, "unique_id"] == "g1_1_2025-12-11"


def test_add_unique_id_missing_columns(monkeypatch):
    df = pd.DataFrame({"TEAM_ID": [2]})
    out = add_unique_id(df)
    # GAME_ID should be placeholder
    assert out.loc[0, "GAME_ID"].startswith("unknown_game")
    # prediction_date should default to today
    assert out.loc[0, "prediction_date"] == str(datetime.date.today())


def test_add_unique_id_type_enforcement():
    df = pd.DataFrame(
        {"GAME_ID": [123], "TEAM_ID": ["5"], "prediction_date": ["2025-12-11"]}
    )
    out = add_unique_id(df)
    # TEAM_ID coerced to int
    assert out.loc[0, "TEAM_ID"] == 5
    # GAME_ID coerced to str
    assert isinstance(out.loc[0, "GAME_ID"], str)


def test_add_unique_id_deduplication():
    df = pd.DataFrame(
        {
            "GAME_ID": ["g1", "g1"],
            "TEAM_ID": [1, 1],
            "prediction_date": ["2025-12-11", "2025-12-11"],
        }
    )
    out = add_unique_id(df)
    # Deduplication should drop duplicate unique_id rows
    assert len(out) == 1
    assert out["unique_id"].is_unique
