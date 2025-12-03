# ============================================================
# File: modeling/predict.py
# Purpose: Generate predictions from trained model
# ============================================================

import pandas as pd
import pickle
import os
from core.logging import setup_logger
from core.config import MODELS_DIR, PREDICTIONS_FILE
from core.exceptions import PipelineError

logger = setup_logger("predict_model")

def predict(features_df, model_file=None):
    if model_file is None:
        model_file = MODELS_DIR / "game_predictor.pkl"

    if not os.path.exists(model_file):
        raise PipelineError("Trained model not found")

    with open(model_file, "rb") as f:
        model = pickle.load(f)

    if not hasattr(model, "predict_proba"):
        raise PipelineError("Model does not support predict_proba")

    # Select only numeric columns
    X = features_df.select_dtypes(include=["number"])

    features_df["pred_home_win_prob"] = model.predict_proba(X)[:, 1]
    features_df.to_csv(PREDICTIONS_FILE, index=False)
    logger.info(f"Predictions saved to {PREDICTIONS_FILE}")
    return features_df