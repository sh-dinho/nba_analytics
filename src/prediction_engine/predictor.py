# ============================================================
# File: src/prediction_engine/predictor.py
# Purpose: Wrap model for predictions with probabilities and labels
# ============================================================

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from scipy.special import expit, softmax

class Predictor:
    def __init__(self, model: BaseEstimator):
        self.model = model

    def _validate_input(self, X):
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise TypeError("Input must be DataFrame or ndarray")
        if X.shape[0] == 0:
            raise ValueError("Input data has zero samples")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._validate_input(X)
        if hasattr(self.model, "predict_proba"):
            return np.asarray(self.model.predict_proba(X))
        if hasattr(self.model, "decision_function"):
            scores = self.model.decision_function(X)
            if scores.ndim == 1:
                probs = expit(scores)
                return np.vstack([1 - probs, probs]).T
            else:
                return softmax(scores, axis=1)
        raise AttributeError("Model does not support probability prediction")

    def predict_label(self, X: pd.DataFrame, threshold=0.5) -> np.ndarray:
        proba = self.predict_proba(X)
        if proba.shape[1] == 2:
            return (proba[:, 1] >= threshold).astype(int)
        return proba.argmax(axis=1)
