# ============================================================
# Path: tests/test_mlflow_setup.py
# Purpose: Unit tests for mlflow_setup.py with metadata + system info + packages
# Project: nba_analysis
# ============================================================

import mlflow
from src.mlflow_setup import configure_mlflow, start_run_with_metadata


def test_start_run_with_packages(monkeypatch, tmp_path):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file:{tmp_path}/mlruns")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "TestExperiment")

    configure_mlflow()
    run = start_run_with_metadata(run_name="test_run")

    data = mlflow.get_run(run.info.run_id).data
    tags = data.tags
    params = data.params

    assert tags["project"] == "nba_analysis"
    assert "timestamp" in tags
    assert "git_commit" in tags
    assert "python_version" in tags
    assert "os" in tags
    assert any(k.startswith("pkg_") for k in params)
