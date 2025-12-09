# ============================================================
# Path: src/prediction_engine/predictor.py
# Filename: predictor.py
# Author: Your Team
# Date: December 2025
# Purpose: Wrapper around trained NBA logistic regression model
# ============================================================

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

class NBAPredictor:
    """
    Wrapper class for NBA logistic regression predictor.
    """

    def __init__(self, model_path: str = None):
        """
        Initialize predictor by loading trained model.

        Args:
            model_path (str): Path to saved model file (.pkl).
        """
        if model_path:
            self.model = joblib.load(model_path)
        else:
            # Default: create a dummy untrained model for CLI usage
            self.model = LogisticRegression(max_iter=1000)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict labels (win/loss).

        Args:
            X (pd.DataFrame): Feature matrix.

        Returns:
            pd.Series: Predicted labels (0 or 1).
        """
        preds = self.model.predict(X)
        return pd.Series(preds, index=X.index)

    def predict_proba(self, X: pd.DataFrame) -> list[float]:
        """
        Predict probabilities of winning.

        Args:
            X (pd.DataFrame): Feature matrix.

        Returns:
            list[float]: List of probabilities for the positive class (win).
        """
        # predict_proba returns shape (n_samples, 2)
        proba = self.model.predict_proba(X)
        # Take probability of class "1" (win)
        return proba[:, 1].tolist()
