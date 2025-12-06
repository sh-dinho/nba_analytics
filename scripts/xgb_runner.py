# ============================================================
# File: scripts/xgb_runner.py
# Purpose: Integrating features into XGBoost model prediction with validation and Kelly/EV
# ============================================================

from typing import List, Dict, Any
from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb

from nba_core.log_config import init_global_logger
from nba_core.exceptions import DataError
from nba_core.config import XGB_ML_MODEL_FILE, XGB_OU_MODEL_FILE
from scripts.utils import expected_value, kelly_fraction  # assumes you have this

logger = init_global_logger("xgb_runner")

def _load_booster(path: Path) -> xgb.Booster | None:
    booster = xgb.Booster()
    try:
        booster.load_model(str(path))
        return booster
    except Exception as e:
        logger.warning(f"⚠️ Failed to load XGB model {path}: {e}")
        return None

xgb_ml = _load_booster(XGB_ML_MODEL_FILE)
xgb_ou = _load_booster(XGB_OU_MODEL_FILE)

def _softmax_row(logits: np.ndarray) -> np.ndarray:
    exp_vals = np.exp(logits - np.max(logits))
    return exp_vals / exp_vals.sum()

def xgb_predict(features_df: pd.DataFrame, use_kelly: bool = True) -> List[Dict[str, Any]]:
    feature_columns = [
        "home_points_avg", "away_points_avg",
        "home_fg_pct_avg", "away_fg_pct_avg",
        "home_rebounds_avg", "away_rebounds_avg",
        "home_implied_prob", "away_implied_prob",
        "home_wins_last_5", "away_wins_last_5",
        "home_points_diff_last_5", "away_points_diff_last_5"
    ]
    missing = set(feature_columns) - set(features_df.columns)
    if missing:
        raise DataError(f"Missing required feature columns: {missing}")

    feature_data = features_df[feature_columns].values
    ml_preds_raw = xgb_ml.predict(xgb.DMatrix(feature_data)) if xgb_ml else []
    ou_preds_raw = xgb_ou.predict(xgb.DMatrix(feature_data)) if xgb_ou else []

    results: List[Dict[str, Any]] = []
    for i, row in enumerate(features_df.itertuples()):
        ml_probs = _softmax_row(np.array(ml_preds_raw[i])) if len(ml_preds_raw) else np.array([0.5, 0.5])
        ou_probs = _softmax_row(np.array(ou_preds_raw[i])) if len(ou_preds_raw) else np.array([0.5, 0.5])

        ev_home = expected_value(ml_probs[1], row.home_implied_prob)
        ev_away = expected_value(ml_probs[0], row.away_implied_prob)
        k_home = kelly_fraction(ml_probs[1], row.home_implied_prob) if use_kelly else None
        k_away = kelly_fraction(ml_probs[0], row.away_implied_prob) if use_kelly else None

        result = {
            "home_team": getattr(row, "home_team", None),
            "away_team": getattr(row, "away_team", None),
            "home_prob": round(ml_probs[1], 4),
            "away_prob": round(ml_probs[0], 4),
            "ou_prediction": "OVER" if ou_probs[1] > ou_probs[0] else "UNDER",
            "home_implied_prob": round(row.home_implied_prob, 4),
            "away_implied_prob": round(row.away_implied_prob, 4),
            "ev_home": round(ev_home, 4),
            "ev_away": round(ev_away, 4),
            "kelly_home": round(k_home, 4) if k_home is not None else None,
            "kelly_away": round(k_away, 4) if k_away is not None else None,
        }
        results.append(result)

    logger.info(f"Predicted {len(results)} games with XGB models")
    return results
