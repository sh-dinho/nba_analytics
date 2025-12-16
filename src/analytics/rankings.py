# ============================================================
# File: src/analytics/rankings.py
# Purpose: Generate rankings and betting recommendations
# ============================================================

import mlflow
import pandas as pd
import xgboost as xgb
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

MODEL_PATH = Path("models/xgb_nba_model.json")


class RankingsManager:
    def __init__(self, mlflow_enabled=True):
        self.mlflow_enabled = mlflow_enabled
        self.model = self.load_model()

    def load_model(self):
        if MODEL_PATH.exists():
            model = xgb.XGBClassifier()
            model.load_model(MODEL_PATH)
            logger.info(f"Loaded ML model from {MODEL_PATH}")
            return model
        else:
            logger.warning(f"ML model not found at {MODEL_PATH}, predictions disabled")
            return None

    def generate_rankings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generates predicted win probabilities"""
        if self.model is None:
            raise ValueError("ML model not loaded")

        # Example: assume df has feature columns starting with 'feat_'
        feature_cols = [c for c in df.columns if c.startswith("feat_")]
        if not feature_cols:
            raise ValueError("No feature columns found for prediction")

        X = df[feature_cols]
        df["predicted_win"] = self.model.predict_proba(X)[:, 1]
        df["predicted_rank"] = df["predicted_win"].rank(ascending=False)

        # MLflow logging
        if self.mlflow_enabled:
            mlflow.log_metric("max_predicted_win", df["predicted_win"].max())

        return df

    def betting_recommendations(
        self, df: pd.DataFrame, win_thr=0.6, acc_thr=0.6
    ) -> dict:
        """
        Returns a dict of teams recommended for betting
        - win_thr: minimum predicted probability
        """
        if "predicted_win" not in df.columns:
            raise ValueError("Predictions not available")

        bet_on = df[df["predicted_win"] >= win_thr]
        return {"bet_on": bet_on}


# Optional: CLI test
if __name__ == "__main__":
    mgr = RankingsManager()
    dummy_df = pd.DataFrame(
        {
            "feat_off_rating": [110, 105],
            "feat_def_rating": [102, 108],
            "TEAM_ABBREVIATION": ["LAL", "BOS"],
        }
    )
    rankings = mgr.generate_rankings(dummy_df)
    print(rankings)
