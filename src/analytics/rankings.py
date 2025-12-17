# ============================================================
# File: src/analytics/rankings.py
# Purpose: Generate rankings and betting recommendations
# ============================================================

import mlflow
import pandas as pd
import xgboost as xgb
from pathlib import Path
import logging

# Initialize the logger
logger = logging.getLogger(__name__)

# Define the path for the trained XGBoost model
MODEL_PATH = Path("models/xgb_nba_model.json")


class RankingsManager:
    def __init__(self, mlflow_enabled=True):
        """
        Initializes the RankingsManager class.

        Args:
            mlflow_enabled (bool): Flag to enable MLflow logging (default is True).
        """
        self.mlflow_enabled = mlflow_enabled
        self.model = self.load_model()

    def load_model(self):
        """
        Loads the trained XGBoost model from the specified path.

        Returns:
            model (xgb.XGBClassifier): Trained XGBoost model.
        """
        if MODEL_PATH.exists():
            model = xgb.XGBClassifier()
            model.load_model(MODEL_PATH)
            logger.info(f"Loaded ML model from {MODEL_PATH}")
            return model
        else:
            logger.warning(f"ML model not found at {MODEL_PATH}, predictions disabled")
            return None

    def generate_rankings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates predicted win probabilities and rankings for teams based on features.

        Args:
            df (pd.DataFrame): DataFrame with feature columns for prediction.

        Returns:
            pd.DataFrame: Updated DataFrame with predicted win probabilities and rankings.
        """
        if self.model is None:
            raise ValueError("ML model not loaded")

        feature_cols = [c for c in df.columns if c.startswith("feat_")]
        if not feature_cols:
            raise ValueError("No feature columns found for prediction")

        X = df[feature_cols]

        # Predict win probabilities using the loaded model
        df["predicted_win"] = self.model.predict_proba(X)[:, 1]

        # Rank teams by predicted win probability
        df["predicted_rank"] = df["predicted_win"].rank(ascending=False)

        # Log maximum predicted win probability to MLflow
        if self.mlflow_enabled:
            mlflow.log_metric("max_predicted_win", df["predicted_win"].max())

        logger.info("Rankings generated successfully.")
        return df

    def betting_recommendations(self, df: pd.DataFrame, win_thr=0.6) -> dict:
        """
        Generates betting recommendations based on predicted win probabilities.

        Args:
            df (pd.DataFrame): DataFrame with predicted win probabilities.
            win_thr (float): Minimum predicted win probability to recommend betting.

        Returns:
            dict: Dictionary with teams recommended for betting.
        """
        if "predicted_win" not in df.columns:
            raise ValueError("Predictions not available")

        # Filter teams with a predicted win probability above the threshold
        bet_on = df[df["predicted_win"] >= win_thr]
        logger.info(f"{len(bet_on)} teams recommended for betting.")
        return {"bet_on": bet_on}
