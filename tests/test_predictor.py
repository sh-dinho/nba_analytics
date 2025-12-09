# ============================================================
# Path: tests/test_predictor.py
# Purpose: Unit tests for src/prediction_engine/predictor.py
# Project: nba_analysis
# ============================================================

import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from src.model_inference.predictor import Predictor


def test_predict_proba_and_label_binary():
    X = pd.DataFrame({"feat1": [0, 1, 0, 1], "feat2": [1, 0, 1, 0]})
    y = np.array([0, 1, 0, 1])
    model = LogisticRegression().fit(X, y)
    predictor = Predictor(model)

    proba = predictor.predict_proba(X)
    assert proba.shape[1] == 2
    labels = predictor.predict_label(X, threshold=0.5)
    assert set(labels).issubset({0, 1})


def test_invalid_input():
    model = LogisticRegression()
    predictor = Predictor(model)
    with pytest.raises(TypeError):
        predictor.predict_proba("not a dataframe")
    with pytest.raises(ValueError):
        predictor.predict_proba(pd.DataFrame())
