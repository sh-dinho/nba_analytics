# Path: tests/test_train_model.py
import os
import pytest
import joblib

import scripts.train_model as train_model

RESULTS_DIR = "results"
MODEL_PATH = train_model.MODEL_PATH
CURVES_FILE = os.path.join(RESULTS_DIR, "eval_curves.csv")
METRICS_FILE = os.path.join(RESULTS_DIR, "eval_metrics.json")

@pytest.mark.parametrize("seasons", [["2021-22"]])
def test_train_and_evaluate_creates_files(seasons):
    # Run training on a single season (smoke test)
    train_model.train_and_evaluate(seasons)

    # Check that model file exists
    assert os.path.exists(MODEL_PATH), "Model file not created"

    # Check that evaluation outputs exist
    assert os.path.exists(CURVES_FILE), "Eval curves file not created"
    assert os.path.exists(METRICS_FILE), "Eval metrics file not created"

    # Load model and ensure it can predict
    model = joblib.load(MODEL_PATH)
    X, y = train_model.build_dataset(seasons)
    preds = model.predict(X)
    assert len(preds) == len(y), "Prediction length mismatch"