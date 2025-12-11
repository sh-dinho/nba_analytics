# ============================================================
# File: src/predictor/predictor.py
# Purpose: Wrapper class for scikit-learn models with proba, labels, batch prediction
# Project: nba_analysis
# Version: 1.4 (named logger, multi-class logging, sparse batch handling)
# ============================================================

import numpy as np
import pandas as pd
import logging
from scipy.special import expit, softmax
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from scipy import sparse

logger = logging.getLogger("predictor.Predictor")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class Predictor:
    def __init__(self, model: BaseEstimator):
        self.model = model

    def _validate_input(self, X):
        if not isinstance(X, (pd.DataFrame, np.ndarray, sparse.spmatrix)):
            raise TypeError(
                "Input must be a pandas DataFrame, NumPy array, or scipy sparse matrix."
            )
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
            raise AttributeError(
                f"Model {type(self.model).__name__} does not support predict_proba or decision_function"
            )

        # Log stats
        if proba.shape[1] == 2:
            p1 = proba[:, 1]
            logger.info("Average win probability: %.3f", p1.mean())
            logger.info("Min: %.3f, Max: %.3f, Std: %.3f", p1.min(), p1.max(), p1.std())
        else:
            logger.info("Multi-class probabilities; logging per-class means.")
            for i in range(proba.shape[1]):
                logger.info("Class %d mean probability: %.3f", i, proba[:, i].mean())
        return proba

    def predict_label(self, X, threshold: float = 0.5) -> np.ndarray:
        self._validate_input(X)
        proba = self.predict_proba(X)
        if proba.shape[1] == 2:
            labels = (proba[:, 1] >= threshold).astype(int)
            logger.info("Predicted win rate: %.3f", labels.mean())
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
            if isinstance(X, pd.DataFrame):
                batch = X.iloc[i : i + batch_size]
            elif sparse.issparse(X):
                batch = X[i : i + batch_size, :]
            else:
                batch = X[i : i + batch_size]
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
            raise ValueError(
                "SVC must be initialized with probability=True to support predict_proba."
            )
        super().__init__(model)
