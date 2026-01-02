from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Prediction Heads
# File: src/model/predict.py
# Author: Sadiq
#
# Description:
#     Production‑grade prediction wrappers that take canonical
#     feature DataFrames and trained models, and return
#     prediction DataFrames for each prediction head.
#     Enhanced with:
#       • model-family–aware prediction logic
#       • validation of game_id/team columns
# ============================================================

import numpy as np
import pandas as pd
from loguru import logger

from src.model.config.model_config import FEATURE_MAP


# ------------------------------------------------------------
# Internal helper
# ------------------------------------------------------------

def _extract_feature_matrix(
    df: pd.DataFrame,
    model_type: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Extract the feature matrix X from a canonical feature DataFrame.

    - Validates required features for the given model_type
    - Ensures stable column ordering
    - Validates presence of game_id and team columns
    """
    if df.empty:
        raise ValueError("Feature DataFrame is empty.")

    # Validate identity columns
    required_identity = ["game_id", "team"]
    missing_identity = [c for c in required_identity if c not in df.columns]
    if missing_identity:
        raise ValueError(f"Missing required identity columns: {missing_identity}")

    # Validate model-specific features
    required_features = FEATURE_MAP[model_type]
    missing_features = [f for f in required_features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required feature columns: {missing_features}")

    X = df[required_features].to_numpy(dtype=float)
    return df, X


# ------------------------------------------------------------
# Model-family–aware prediction helpers
# ------------------------------------------------------------

def _predict_proba_safe(model, X: np.ndarray) -> np.ndarray:
    """
    Predict probabilities in a model-family–aware way.
    Supports:
        - XGBoost
        - LightGBM
        - sklearn LogisticRegression
        - CalibratedClassifierCV
    """
    # Standard sklearn API
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] == 2:
            return proba[:, 1]
        # Some models return only positive class probability
        if proba.ndim == 1:
            return proba

    # LightGBM booster
    if hasattr(model, "predict") and not hasattr(model, "predict_proba"):
        preds = model.predict(X)
        # LightGBM returns raw scores unless configured otherwise
        # Clip to [0,1] as a fallback
        return np.clip(preds, 0.0, 1.0)

    raise AttributeError("Model does not support probability prediction.")


def _predict_regression_safe(model, X: np.ndarray) -> np.ndarray:
    """
    Predict regression outputs in a model-family–aware way.
    Supports:
        - XGBoost
        - LightGBM
        - sklearn regressors
    """
    if hasattr(model, "predict"):
        return model.predict(X)

    raise AttributeError("Model does not support regression prediction.")


# ------------------------------------------------------------
# Moneyline (win probability)
# ------------------------------------------------------------

def predict_moneyline(features: pd.DataFrame, model) -> pd.DataFrame:
    df, X = _extract_feature_matrix(features, model_type="moneyline")

    proba = _predict_proba_safe(model, X)
    proba = np.clip(proba.astype(float), 0.0, 1.0)

    out = pd.DataFrame(
        {
            "game_id": df["game_id"].astype(str),
            "team": df["team"].astype(str),
            "model_type": "moneyline",
            "win_probability": proba,
        }
    )

    logger.info(f"predict_moneyline: produced {len(out)} rows")
    return out


# ------------------------------------------------------------
# Totals (predicted total points)
# ------------------------------------------------------------

def predict_totals(features: pd.DataFrame, model) -> pd.DataFrame:
    df, X = _extract_feature_matrix(features, model_type="totals")

    preds = _predict_regression_safe(model, X).astype(float)

    out = pd.DataFrame(
        {
            "game_id": df["game_id"].astype(str),
            "team": df["team"].astype(str),
            "model_type": "totals",
            "predicted_total_points": preds,
        }
    )

    logger.info(f"predict_totals: produced {len(out)} rows")
    return out


# ------------------------------------------------------------
# Spread (predicted scoring margin)
# ------------------------------------------------------------

def predict_spread(features: pd.DataFrame, model) -> pd.DataFrame:
    df, X = _extract_feature_matrix(features, model_type="spread")

    preds = _predict_regression_safe(model, X).astype(float)

    out = pd.DataFrame(
        {
            "game_id": df["game_id"].astype(str),
            "team": df["team"].astype(str),
            "model_type": "spread",
            "predicted_margin": preds,
        }
    )

    logger.info(f"predict_spread: produced {len(out)} rows")
    return out


# ------------------------------------------------------------
# Thresholding helper
# ------------------------------------------------------------

def apply_threshold(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    if "win_probability" not in df.columns:
        raise ValueError("win_probability column missing.")

    out = df.copy()
    out["predicted_win"] = (out["win_probability"] >= threshold).astype(int)
    return out
