import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from scipy.special import softmax, expit

class Predictor:
    """
    Wraps a scikit-learn model to provide probability and label predictions.
    Supports predict_proba, decision_function fallback, and thresholded labels.
    """

    def __init__(self, model: BaseEstimator):
        self.model = model

    def _validate_input(self, X):
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise TypeError("Input must be a pandas DataFrame or NumPy array.")
        if X.shape[0] == 0:
            raise ValueError("Input data has zero samples.")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities. Falls back to decision_function if predict_proba is unavailable.
        """
        self._validate_input(X)

        if hasattr(self.model, "predict_proba"):
            return np.asarray(self.model.predict_proba(X))

        if hasattr(self.model, "decision_function"):
            scores = self.model.decision_function(X)
            if scores.ndim == 1:  # binary classification
                probs = expit(scores)
                return np.vstack([1 - probs, probs]).T
            else:  # multi-class classification
                return softmax(scores, axis=1)

        raise AttributeError("Model does not support predict_proba or decision_function")

    def predict_label(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels. For binary classification, allows thresholding on probabilities.
        """
        self._validate_input(X)
        proba = self.predict_proba(X)

        if proba.shape[1] == 2:  # binary classification
            return (proba[:, 1] >= threshold).astype(int)

        # multi-class: argmax of probabilities
        return proba.argmax(axis=1)
