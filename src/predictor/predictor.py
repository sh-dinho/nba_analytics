# ============================================================
# File: src/predictor/predictor.py
# Purpose: A wrapper class for scikit-learn models providing
#          probability predictions, label predictions with
#          thresholding, and batch prediction for large datasets.
#          Also includes special handling for Logistic Regression
#          and Support Vector Classification (SVC).
#
# Key Features:
# 1. Support for `predict_proba` and `decision_function` fallbacks.
# 2. Predicts class labels based on customizable probability thresholds.
# 3. Batch processing of large datasets to improve performance.
# 4. Special handling for Logistic Regression to use log probabilities.
# 5. Ensures SVC models are trained with `probability=True` to output probabilities.
# 6. Model metadata retrieval for logging or debugging purposes.
#
# Dependencies:
# - numpy
# - pandas
# - scipy
# - scikit-learn
# ============================================================

import numpy as np
import pandas as pd
import logging
from scipy.special import expit, softmax
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from scipy import sparse

class Predictor:
    def __init__(self, model: BaseEstimator, log_level="INFO", log_name="Predictor"):
        self.model = model
        logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO),
                            format="%(asctime)s [%(levelname)s] %(message)s")
        self.logger = logging.getLogger(log_name)

    def _validate_input(self, X):
        if not isinstance(X, (pd.DataFrame, np.ndarray, sparse.spmatrix)):
            raise TypeError("Input must be a pandas DataFrame, NumPy array, or scipy sparse matrix.")
        if X.shape[0] == 0:
            raise ValueError("Input data has zero samples.")

    def predict_proba(self, X) -> np.ndarray:
        self._validate_input(X)
        if hasattr(self.model, "predict_proba"):
            proba = np.asarray(self.model.predict_proba(X))
        elif hasattr(self.model, "decision_function"):
            scores = self.model.decision_function(X)
            if scores.ndim == 1:  # binary
                p1 = expit(scores)
                proba = np.column_stack([1 - p1, p1])
            else:  # multi-class
                proba = softmax(scores, axis=1)
        else:
            raise AttributeError(f"Model {type(self.model).__name__} does not support predict_proba or decision_function")

        # Log stats
        if proba.shape[1] == 2:
            p1 = proba[:, 1]
            self.logger.info(f"Average win probability: {p1.mean():.3f}")
            self.logger.info(f"Min: {p1.min():.3f}, Max: {p1.max():.3f}, Std: {p1.std():.3f}")
        return proba

    def predict_label(self, X, threshold: float = 0.5) -> np.ndarray:
        self._validate_input(X)
        proba = self.predict_proba(X)
        if proba.shape[1] == 2:
            labels = (proba[:, 1] >= threshold).astype(int)
            self.logger.info(f"Predicted win rate: {labels.mean():.3f}")
            return labels
        return proba.argmax(axis=1)

    def fit(self, X, y) -> "Predictor":
        if hasattr(self.model, "fit"):
            self.model.fit(X, y)
            return self
        raise AttributeError("The model does not have a fit method.")

    def batch_predict_proba(self, X, batch_size: int = 1000) -> np.ndarray:
        self._validate_input(X)
        n_samples = X.shape[0]
        all_probs = []
        for i in range(0, n_samples, batch_size):
            batch = X.iloc[i:i+batch_size] if isinstance(X, pd.DataFrame) else X[i:i+batch_size]
            all_probs.append(self.predict_proba(batch))
        return np.vstack(all_probs)

    def get_model_name(self) -> str:
        return self.model.__class__.__name__

    def get_params(self) -> dict:
        return self.model.get_params()


class LogisticRegressionPredictor(Predictor):
    def __init__(self, model: LogisticRegression):
        if not isinstance(model, LogisticRegression):
            raise TypeError("Expected a LogisticRegression model.")
        super().__init__(model)

    def predict_proba(self, X) -> np.ndarray:
        self._validate_input(X)
        if hasattr(self.model, "predict_log_proba"):
            log_proba = self.model.predict_log_proba(X)
            return np.exp(log_proba)
        return super().predict_proba(X)


class SVCWithProbabilities(Predictor):
    def __init__(self, model: SVC):
        if not isinstance(model, SVC):
            raise TypeError("Expected an SVC model.")
        if not model.probability:
            raise ValueError("SVC must be initialized with probability=True to support predict_proba.")
        super().__init__(model)
