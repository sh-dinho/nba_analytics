# ============================================================
# File: tests/test_io.py
# Purpose: Validate DataFrame I/O utilities (load, save, read_or_create)
# Project: nba_analysis
# ============================================================

import os
import pandas as pd
import pytest

from src.utils.io import load_dataframe, save_dataframe, read_or_create


def test_save_and_load_csv(tmp_path):
    df = pd.DataFrame({"A": [1, 2], "B": ["x", "y"]})
    path = tmp_path / "test.csv"

    # Save and load
    save_dataframe(df, str(path))
    loaded = load_dataframe(str(path))

    pd.testing.assert_frame_equal(df, loaded)


def test_save_and_load_parquet(tmp_path):
    df = pd.DataFrame({"A": [1, 2], "B": ["x", "y"]})
    path = tmp_path / "test.parquet"

    save_dataframe(df, str(path))
    loaded = load_dataframe(str(path))

    pd.testing.assert_frame_equal(df, loaded)


def test_load_dataframe_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_dataframe("nonexistent.csv")


def test_save_dataframe_invalid_type(tmp_path):
    path = tmp_path / "bad.csv"
    with pytest.raises(TypeError):
        save_dataframe([1, 2, 3], str(path))


def test_read_or_create_creates_file(tmp_path):
    default_df = pd.DataFrame({"GAME_ID": [], "TEAM_ID": [], "PTS": []})
    path = tmp_path / "games.csv"

    # File does not exist, should create
    df = read_or_create(str(path), default_df)
    assert df.equals(default_df)
    assert os.path.exists(path)


def test_read_or_create_loads_existing(tmp_path):
    df = pd.DataFrame({"A": [1], "B": [2]})
    path = tmp_path / "existing.csv"

    df.to_csv(path, index=False)

    loaded = read_or_create(str(path), pd.DataFrame())
    pd.testing.assert_frame_equal(df, loaded)
