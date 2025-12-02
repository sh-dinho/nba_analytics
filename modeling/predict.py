# modeling/predict.py
import pandas as pd
import pickle
from core.log_config import setup_logger
from core.paths import MODELS_DIR
import os

logger = setup_logger("predict_model")

def predict(features_df, model_file=None):
    if model_file is None:
        model_file = os.path.join(MODELS_DIR, "game_predictor.pkl")
    
    if not os.path.exists(model_file):
        raise FileNotFoundError("Trained model not found")
    
    with open(model_file, "rb") as f:
        model = pickle.load(f)
    
    X = features_df
    features_df["pred_home_win_prob"] = model.predict_proba(X)[:,1]
    logger.info("Predictions generated")
    return features_df
