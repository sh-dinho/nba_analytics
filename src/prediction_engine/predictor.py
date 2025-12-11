# ============================================================
# File: src/prediction_engine/predictor.py
# Purpose: Load trained model and run predictions
# Project: nba_analysis
# Version: 2.0 (schedule-based features, safe preprocessing)
# ============================================================

import logging
import joblib
import pandas as pd

logger = logging.getLogger("prediction_engine.predictor")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class Predictor:
    def __init__(self, model_path: str):
        """Load a trained model from disk."""
        try:
            self.model = joblib.load(model_path)
            logger.info("Loaded model from %s", model_path)
        except Exception as e:
            logger.error("Failed to load model: %s", e)
            raise

    def _prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure features are in the right format for prediction.
        Handles categorical TEAM_NAME and numeric HOME_GAME/DAYS_SINCE_GAME.
        """
        df = X.copy()

        # TEAM_NAME categorical encoding
        if "TEAM_NAME" in df.columns:
            df["TEAM_NAME"] = df["TEAM_NAME"].astype("category").cat.codes

        # HOME_GAME numeric
        if "HOME_GAME" in df.columns:
            df["HOME_GAME"] = pd.to_numeric(df["HOME_GAME"], errors="coerce").fillna(0)

        # DAYS_SINCE_GAME numeric
        if "DAYS_SINCE_GAME" in df.columns:
            df["DAYS_SINCE_GAME"] = pd.to_numeric(
                df["DAYS_SINCE_GAME"], errors="coerce"
            ).fillna(0)

        return df

    def predict_proba(self, X: pd.DataFrame):
        """Return win probability predictions."""
        df = self._prepare_features(X)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(df)[:, 1]
        else:
            logger.warning(
                "Model does not support predict_proba, using predict instead."
            )
            return self.model.predict(df)

    def predict_label(self, X: pd.DataFrame):
        """Return win/loss label predictions."""
        df = self._prepare_features(X)
        return self.model.predict(df)
