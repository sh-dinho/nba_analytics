# ============================================================
# Path: tests/test_pipeline.py
# Purpose: Unit tests for NBA analytics pipeline model selection
# Version: 1.1
# ============================================================

import os
import yaml
import pytest
from run_pipeline import load_config, main

# -----------------------------
# Helper: Write config file
# -----------------------------
def write_config(model_type: str):
    config = {"model_type": model_type}
    with open("config.yaml", "w") as f:
        yaml.safe_dump(config, f)

# -----------------------------
# Test: Logistic Regression selection
# -----------------------------
def test_pipeline_logreg(monkeypatch):
    write_config("logreg")

    # Monkeypatch train_logreg to avoid heavy training
    import src.model_training.train_logreg as train_logreg
    monkeypatch.setattr(train_logreg, "train_logreg", lambda **kwargs: {"metrics": {"accuracy": 0.95}})

    main()  # Run pipeline
    # Assert config loaded correctly
    cfg = load_config()
    assert cfg["model_type"] == "logreg"

# -----------------------------
# Test: XGBoost selection
# -----------------------------
def test_pipeline_xgb(monkeypatch):
    write_config("xgb")

    # Monkeypatch train_xgb to avoid heavy training
    import src.model_training.train_xgb as train_xgb
    monkeypatch.setattr(train_xgb, "train_xgb", lambda **kwargs: {"metrics": {"logloss": 0.25}, "model_path": "models/nba_xgb.json"})

    # Monkeypatch SHAP analysis to avoid plotting
    import src.interpretability.shap_analysis as shap_analysis
    monkeypatch.setattr(shap_analysis, "run_shap", lambda *args, **kwargs: None)

    main()  # Run pipeline
    # Assert config loaded correctly
    cfg = load_config()
    assert cfg["model_type"] == "xgb"

# -----------------------------
# Test: Invalid model_type
# -----------------------------
def test_pipeline_invalid(monkeypatch, caplog):
    write_config("invalid_model")

    # Run pipeline and capture logs
    main()
    assert "Unknown model_type" in caplog.text
