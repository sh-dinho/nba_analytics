# ============================================================
# File: scripts/xgb_runner.py
# Purpose: Integrating features into XGBoost model prediction
# ============================================================

from typing import List, Dict, Any
import pandas as pd
import numpy as np
import xgboost as xgb
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

def _softmax_row(logits: np.ndarray) -> np.ndarray:
    """Apply softmax to a 1D array of logits."""
    exp_vals = np.exp(logits - np.max(logits))
    return exp_vals / exp_vals.sum()

def xgb_predict(features_df: pd.DataFrame, use_kelly: bool = True) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    feature_columns = [
        "home_points_avg", "away_points_avg",
        "home_fg_pct_avg", "away_fg_pct_avg",
        "home_rebounds_avg", "away_rebounds_avg",
        "home_implied_prob", "away_implied_prob",
        "home_wins_last_5", "away_wins_last_5",
        "home_points_diff_last_5", "away_points_diff_last_5"
    ]
    feature_data = features_df[feature_columns].values

    # Predict with XGBoost
    ml_preds_raw = xgb_ml.predict(xgb.DMatrix(feature_data))
    ou_preds_raw = xgb_ou.predict(xgb.DMatrix(feature_data))

    for i, row in enumerate(features_df.itertuples()):
        ml_probs = _softmax_row(np.array(ml_preds_raw[i]))
        ou_probs = _softmax_row(np.array(ou_preds_raw[i]))

        # Expected value & Kelly fraction (optional)
        ev_home = expected_value(ml_probs[1], row.home_implied_prob)
        ev_away = expected_value(ml_probs[0], row.away_implied_prob)

        kelly_home = kelly_fraction(ml_probs[1], row.home_implied_prob) if use_kelly else None
        kelly_away = kelly_fraction(ml_probs[0], row.away_implied_prob) if use_kelly else None

        result = {
            "home_team": row.home_team,
            "away_team": row.away_team,
            "home_prob": round(ml_probs[1], 4),
            "away_prob": round(ml_probs[0], 4),
            "ou_prediction": "OVER" if ou_probs[1] > ou_probs[0] else "UNDER",
            "home_implied_prob": round(row.home_implied_prob, 4),
            "away_implied_prob": round(row.away_implied_prob, 4),
            "ev_home": round(ev_home, 4),
            "ev_away": round(ev_away, 4),
            "kelly_home": round(kelly_home, 4) if kelly_home is not None else None,
            "kelly_away": round(kelly_away, 4) if kelly_away is not None else None,
        }
        results.append(result)

    return results