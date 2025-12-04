# ============================================================
# File: scripts/xgb_runner.py
# Purpose: Integrating features into XGBoost model prediction
# ============================================================

from typing import List, Tuple, Any, Dict, Optional
import xgboost as xgb
import numpy as np
from scripts.utils import expected_value, kelly_fraction
from core.config import (
    XGB_ML_MODEL_FILE,
    XGB_OU_MODEL_FILE,
)

# Load XGBoost models
xgb_ml = xgb.Booster()
xgb_ml.load_model(str(XGB_ML_MODEL_FILE))

xgb_ou = xgb.Booster()
xgb_ou.load_model(str(XGB_OU_MODEL_FILE))

def xgb_predict(features_df: pd.DataFrame, use_kelly: bool = True) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    
    # Extract the features from the dataframe (excluding non-feature columns)
    feature_columns = ['home_points_avg', 'away_points_avg', 'home_fg_pct_avg', 'away_fg_pct_avg',
                       'home_rebounds_avg', 'away_rebounds_avg', 'home_implied_prob', 'away_implied_prob',
                       'home_wins_last_5', 'away_wins_last_5', 'home_points_diff_last_5', 'away_points_diff_last_5']
    feature_data = features_df[feature_columns].values

    # Predict with XGBoost
    ml_preds_raw = xgb_ml.predict(xgb.DMatrix(feature_data))
    ou_preds_raw = xgb_ou.predict(xgb.DMatrix(feature_data))

    for i, row in enumerate(features_df.itertuples()):
        ml_probs = _softmax_row(ml_preds_raw[i])
        ou_probs = _softmax_row(ou_preds_raw[i])

        result = {
            "home_team": row.home_team,
            "away_team": row.away_team,
            "home_prob": round(ml_probs[1], 4),
            "away_prob": round(ml_probs[0], 4),
            "ou_prediction": "OVER" if ou_probs[1] > ou_probs[0] else "UNDER",
            "home_implied_prob": round(row.home_implied_prob, 4),
            "away_implied_prob": round(row.away_implied_prob, 4),
            # Additional details here as needed
        }
        results.append(result)
    
    return results
