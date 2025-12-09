# ============================================================
# Path: tests/test_io.py
# Purpose: Unit tests for src/utils/io.py
# Project: nba_analysis
# ============================================================

import pandas as pd
from src.utils.io import safe_save, safe_load, safe_delete, safe_exists


def test_safe_exists_and_delete(tmp_path):
    df = pd.DataFrame({"a": [1]})
    fpath = tmp_path / "exists.csv"
    safe_save(df, fpath)

    # File should exist
    assert safe_exists(fpath)

    # Delete file
    safe_delete(fpath)

    # File should not exist
    assert not safe_exists(fpath)
