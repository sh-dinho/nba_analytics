# ============================================================
# Path: tests/test_validation.py
# Purpose: Unit tests for src/utils/validation.py
# Project: nba_analysis
# ============================================================

import pytest
from src.utils.validation import validate_config


def test_validate_config_valid(tmp_path):
    cfg = {
        "season": "2025",
        "output_dir": tmp_path,
        "stat": "points",
        "lineup": ["1234567","2345678","3456789","4567890","5678901"]
    }
    validated = validate_config(cfg, ["season","output_dir","stat","lineup"])
    assert validated["season"] == "2024-25"
    assert validated["stat"] == "points"
    assert len(validated["lineup"]) == 5


def test_validate_config_missing_key(tmp_path):
    cfg = {"season": "2025"}
    with pytest.raises(ValueError):
        validate_config(cfg, ["season","output_dir"])
