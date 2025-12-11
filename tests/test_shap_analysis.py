# ============================================================
# File: tests/test_shap_analysis.py
# Purpose: Unit tests for interpretability.shap_analysis.run_shap
# Project: nba_analysis
# ============================================================

import os
import tempfile
import pandas as pd
import numpy as np
import joblib
import pytest

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import src.interpretability.shap_analysis as shap_analysis


@pytest.fixture
def dummy_pipeline(tmp_path):
    """Create a simple sklearn pipeline with XGBClassifier and save it."""
    X = pd.DataFrame(
        {
            "f1": np.random.randn(20),
            "f2": np.random.randn(20),
            "target": np.random.randint(0, 2, 20),
        }
    )
    y = X["target"]
    X_train = X.drop(columns=["target"])

    clf = XGBClassifier(
        n_estimators=5, max_depth=2, use_label_encoder=False, eval_metric="logloss"
    )
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", clf),
        ]
    )
    pipeline.fit(X_train, y)

    model_path = tmp_path / "pipeline.pkl"
    joblib.dump(pipeline, model_path)

    # Save dataset as CSV
    cache_file = tmp_path / "cache.csv"
    X.to_csv(cache_file, index=False)

    return str(model_path), str(cache_file)


@pytest.fixture
def mock_mlflow(monkeypatch):
    """Mock MLflow functions to capture calls."""
    calls = {"artifacts": [], "metrics": {}, "params": {}}

    class DummyRun:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(shap_analysis.mlflow, "start_run", lambda **kwargs: DummyRun())
    monkeypatch.setattr(
        shap_analysis.mlflow,
        "log_artifact",
        lambda path, artifact_path=None: calls["artifacts"].append(
            (path, artifact_path)
        ),
    )
    monkeypatch.setattr(
        shap_analysis.mlflow,
        "log_metric",
        lambda k, v: calls["metrics"].__setitem__(k, v),
    )
    monkeypatch.setattr(
        shap_analysis.mlflow,
        "log_param",
        lambda k, v: calls["params"].__setitem__(k, v),
    )

    return calls


def test_run_shap_creates_plots_and_logs(dummy_pipeline, mock_mlflow, tmp_path):
    model_path, cache_file = dummy_pipeline
    out_dir = tmp_path / "out"

    shap_analysis.run_shap(model_path, cache_file, out_dir=str(out_dir), top_n=2)

    # Check summary plot exists
    summary_path = out_dir / "shap_summary.png"
    assert summary_path.exists()

    # Check dependence plots exist
    dep_files = list(out_dir.glob("shap_dependence_*.png"))
    assert len(dep_files) == 2

    # Check MLflow artifacts logged
    assert any("shap_summary.png" in a[0] for a in mock_mlflow["artifacts"])
    assert any("shap_dependence_" in a[0] for a in mock_mlflow["artifacts"])

    # Check metrics logged for top features
    assert any(k.startswith("shap_mean_abs_") for k in mock_mlflow["metrics"].keys())


def test_run_shap_invalid_step_raises(dummy_pipeline):
    model_path, cache_file = dummy_pipeline
    with pytest.raises(KeyError):
        shap_analysis.run_shap(model_path, cache_file, clf_step="not_a_step")


def test_run_shap_invalid_cache_format(dummy_pipeline, tmp_path):
    model_path, _ = dummy_pipeline
    bad_file = tmp_path / "cache.txt"
    bad_file.write_text("dummy")
    with pytest.raises(ValueError):
        shap_analysis.run_shap(model_path, str(bad_file))
