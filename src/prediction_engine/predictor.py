# ============================================================
# File: src/prediction_engine/predictor.py
# Purpose: Generic predictor wrapper with logging for scikit-learn style models
# Project: nba_analysis
# Version: 1.3 (adds logging hooks for probability stats and win rate)
# ============================================================

import numpy as np
import logging
from scipy.special import expit, softmax

class Predictor:
    def __init__(self, model, log_level="INFO", log_name="Predictor"):
        self.model = model
        logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO),
                            format="%(asctime)s [%(levelname)s] %(message)s")
        self.logger = logging.getLogger(log_name)

    def _predict(self, X, output_type="label", threshold=0.5):
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)
        elif hasattr(self.model, "decision_function"):
            scores = self.model.decision_function(X)
            if scores.ndim > 1:  # multi-class
                proba = softmax(scores, axis=1)
            else:  # binary
                p1 = expit(scores)
                proba = np.column_stack([1 - p1, p1])
        else:
            raise AttributeError(f"Model {type(self.model).__name__} does not support prediction")

        if output_type == "label":
            labels = (proba[:, 1] >= threshold).astype(int)
            win_rate = labels.mean()
            self.logger.info(f"Predicted win rate: {win_rate:.3f}")
            return labels

        # Log probability stats
        p1 = proba[:, 1] if proba.shape[1] > 1 else proba.ravel()
        self.logger.info(f"Average win probability: {p1.mean():.3f}")
        self.logger.info(f"Min: {p1.min():.3f}, Max: {p1.max():.3f}, Std: {p1.std():.3f}")
        return proba

    def predict(self, X, output_type="label", threshold=0.5):
        return self._predict(X, output_type=output_type, threshold=threshold)
