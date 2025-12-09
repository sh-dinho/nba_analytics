# ============================================================
# Path: tests/test_predictor_cli.py
# Filename: test_predictor_cli.py
# Author: Your Team
# Date: December 2025
# Purpose: Tests for NBAPredictor CLI interface
# ============================================================

import pytest
import pandas as pd
from click.testing import CliRunner
from src.prediction_engine.predictor_cli import cli

@pytest.fixture
def runner():
    return CliRunner()

def test_cli_proba_output(runner, tmp_path):
    # Run CLI with --proba option
    result = runner.invoke(cli, ["--proba", "--limit", "1"])
    assert result.exit_code == 0

    # Parse output into DataFrame
    df = pd.read_json(result.output)

    # Ensure probability column exists
    assert "proba" in df.columns

    # Access probability safely
    proba = df["proba"].iloc[0]
    assert 0 <= proba <= 1

def test_cli_label_output(runner, tmp_path):
    # Run CLI with --label option
    result = runner.invoke(cli, ["--label", "--limit", "1"])
    assert result.exit_code == 0

    df = pd.read_json(result.output)

    # Ensure label column exists
    assert "label" in df.columns

    label = df["label"].iloc[0]
    assert label in [0, 1]

def test_cli_with_tags(runner, tmp_path):
    # Run CLI with --tags option
    result = runner.invoke(cli, ["--tags", "--limit", "1"])
    assert result.exit_code == 0

    df = pd.read_json(result.output)

    # Ensure tags column exists
    assert "tags" in df.columns

    tags = df["tags"].iloc[0]
    assert isinstance(tags, list)
