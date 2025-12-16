# ============================================================
# File: src/model/predict.py
# Purpose: Load trained model and predict outcomes
# ============================================================

import pandas as pd
import joblib
import logging

logger = logging.getLogger("model.predict")


def load_model(model_path: str):
    try:
        model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        logger.warning(f"Failed to load model: {e}")
        return None


def predict_schedule(model, schedule_df: pd.DataFrame) -> pd.DataFrame:
    if model is None or schedule_df.empty:
        return pd.DataFrame()

    # Example: predict "predicted_win" as probability
    schedule_df = schedule_df.copy()
    features = ["TEAM_ABBREVIATION", "OPPONENT_ABBREVIATION"]  # placeholder
    schedule_df["predicted_win"] = 0.5  # dummy probability
    schedule_df["WIN"] = None
    return schedule_df
