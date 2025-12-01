# app/predictor.py (Updated with Data Drift Check)
import logging
import pandas as pd
import joblib
import json
import numpy as np

from nba_analytics_core.data import engineer_features
from nba_analytics_core.fetch_games import get_todays_games
from config import THRESHOLD # Using config for threshold

def load_models():
    clf = joblib.load("models/classification_model.pkl")
    reg = joblib.load("models/regression_model.pkl")
    return clf, reg

def check_data_drift(X_live: pd.DataFrame, deviation_threshold: float = 0.1):
    """Checks live feature set against historical stats for drift."""
    try:
        with open("artifacts/feature_stats.json", 'r') as f:
            historical_stats = json.load(f)
            hist_mean = pd.Series(historical_stats['mean'])
            # hist_std = pd.Series(historical_stats['std']) # Only checking mean for simplicity
    except FileNotFoundError:
        logging.warning("Historical feature stats not found. Skipping data drift check.")
        return

    live_mean = X_live.mean()
    
    # Calculate percentage deviation for mean (relative to hist_mean). Adding 1e-6 to avoid division by zero.
    mean_deviation = (abs(live_mean - hist_mean) / (hist_mean.replace(0, np.nan) + 1e-6))
    
    # Filter for features that exceed the deviation threshold
    drift_features = mean_deviation[mean_deviation > deviation_threshold]
    
    if not drift_features.empty:
        logging.warning(
            f"🚨 DATA DRIFT DETECTED IN {len(drift_features)} FEATURES! "
            "Model prediction reliability may be compromised. "
            "Top drifting features (by mean deviation):"
        )
        logging.warning(drift_features.sort_values(ascending=False).to_string())
    else:
        logging.info("✔ Data drift check passed. Live features align with historical statistics.")


def predict_todays_games():
    games = get_todays_games()
    if games.empty:
        logging.info("No games found for today.")
        return pd.DataFrame()

    df_features = engineer_features(games)
    clf, reg = load_models()

    # Prepare feature matrix X by dropping targets
    X = df_features.drop(["home_win", "total_points"], axis=1, errors='ignore')
    
    # --- Run Data Drift Check ---
    check_data_drift(X)
    
    # Log feature importances 
    try:
        if hasattr(clf, 'feature_importances_'):
            feature_names = X.columns
            importances = pd.Series(clf.feature_importances_, index=feature_names)
            top_features = importances.sort_values(ascending=False).head(10)
            logging.info("Top 10 Feature Importances for Classifier:")
            logging.info("\n" + top_features.to_string())
        else:
            logging.warning("Feature importance attribute missing on the classifier model.")
    except Exception as e:
        logging.error(f"Failed to log feature importance: {e}")
    
    # Generate predictions
    df_features["pred_home_win_prob"] = clf.predict_proba(X)[:, 1]
    df_features["pred_total_points"] = reg.predict(X)

    df_pred = games.merge(
        df_features[["game_id", "pred_home_win_prob", "pred_total_points"]],
        on="game_id"
    )
    return df_pred