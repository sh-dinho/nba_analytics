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
# ============================================================

import numpy as np
import pandas as pd
from loguru import logger

from src.model.config.model_config import FEATURES


# ------------------------------------------------------------
# Internal helper
# ------------------------------------------------------------

def _extract_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Extract the feature matrix X from a canonical feature DataFrame.

    - Validates that all required model features exist
    - Ensures stable column ordering
    - Drops identity columns automatically
    """
    if df.empty:
        raise ValueError("Feature DataFrame is empty.")

    identity_cols = {"game_id", "team", "opponent"}

    # Validate required features
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    # Stable ordering
    feature_cols = FEATURES

    X = df[feature_cols].to_numpy(dtype=float)
    return df, X


# ------------------------------------------------------------
# Moneyline (win probability)
# ------------------------------------------------------------

def predict_moneyline(features: pd.DataFrame, model) -> pd.DataFrame:
    """
    Predict win probability using a classification model
    that implements predict_proba().
    """
    df, X = _extract_feature_matrix(features)

    if not hasattr(model, "predict_proba"):
        raise AttributeError("Model does not implement predict_proba().")

    proba = model.predict_proba(X)[:, 1].astype(float)

    # Safety: clip probabilities
    proba = np.clip(proba, 0.0, 1.0)

    out = pd.DataFrame(
        {
            "game_id": df["game_id"].astype(str),
            "team": df["team"].astype(str),
            "win_probability": proba,
        }
    )

    logger.info(f"predict_moneyline: produced {len(out)} rows")
    return out


# ------------------------------------------------------------
# Totals (predicted total points)
# ------------------------------------------------------------

def predict_totals(features: pd.DataFrame, model) -> pd.DataFrame:
    """
    Predict total points using a regression model.
    """
    df, X = _extract_feature_matrix(features)

    if not hasattr(model, "predict"):
        raise AttributeError("Model does not implement predict().")

    preds = model.predict(X).astype(float)

    out = pd.DataFrame(
        {
            "game_id": df["game_id"].astype(str),
            "team": df["team"].astype(str),
            "predicted_total_points": preds,
        }
    )

    logger.info(f"predict_totals: produced {len(out)} rows")
    return out


# ------------------------------------------------------------
# Spread (predicted scoring margin)
# ------------------------------------------------------------

def predict_spread(features: pd.DataFrame, model) -> pd.DataFrame:
    """
    Predict scoring margin using a regression model.
    """
    df, X = _extract_feature_matrix(features)

    if not hasattr(model, "predict"):
        raise AttributeError("Model does not implement predict().")

    preds = model.predict(X).astype(float)

    out = pd.DataFrame(
        {
            "game_id": df["game_id"].astype(str),
            "team": df["team"].astype(str),
            "predicted_margin": preds,
        }
    )

    logger.info(f"predict_spread: produced {len(out)} rows")
    return out


# ------------------------------------------------------------
# Thresholding helper
# ------------------------------------------------------------

def apply_threshold(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    Apply a win/loss threshold to win_probability.
    """
    if "win_probability" not in df.columns:
        raise ValueError("win_probability column missing.")

    df = df.copy()
    df["predicted_win"] = (df["win_probability"] >= threshold).astype(int)
    return df