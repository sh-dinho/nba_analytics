# ============================================================
# 🏀 NBA Analytics v3
# Module: Spread Prediction Pipeline (ATS)
# File: src/model/predict_spread.py
# Author: Sadiq
#
# Description:
#     Loads the latest spread model from the registry,
#     builds home-team features for a target date, and
#     generates predicted_margin for each game.
#
#     Cleaned-up version:
#       - Consistent with moneyline & totals predict.py
#       - Uses unified registry
#       - Fully typed and logged
#       - Backwards-compatible helper alias
# ============================================================

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Tuple, Any

import pandas as pd
from loguru import logger

from src.config.paths import (
    LONG_SNAPSHOT,
    SCHEDULE_SNAPSHOT,
    MODEL_REGISTRY_DIR,
    DATA_DIR,
)
from src.features.builder import FeatureBuilder, FeatureConfig
from src.model.registry import load_model_and_metadata

SPREAD_PRED_DIR = DATA_DIR / "predictions_spread"
SPREAD_PRED_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Data loading
# ------------------------------------------------------------


def _load_schedule(pred_date: date) -> pd.DataFrame:
    df = pd.read_parquet(SCHEDULE_SNAPSHOT)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    todays = df[df["date"] == pred_date].copy()

    if todays.empty:
        logger.warning(f"[Spread] No games scheduled for {pred_date}.")
    return todays


def _load_long() -> pd.DataFrame:
    df = pd.read_parquet(LONG_SNAPSHOT)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


# ------------------------------------------------------------
# Feature building
# ------------------------------------------------------------


def _build_prediction_features(pred_date: date, feature_version: str) -> pd.DataFrame:
    df_long = _load_long()
    df_hist = df_long[df_long["date"] <= pred_date].copy()

    if df_hist.empty:
        logger.warning(f"[Spread] No historical data available up to {pred_date}.")
        return pd.DataFrame()

    fb = FeatureBuilder(config=FeatureConfig(version=feature_version))
    features = fb.build_from_long(df_hist)

    # Add is_home flag
    features = features.merge(
        df_long[["game_id", "team", "date", "is_home"]],
        on=["game_id", "team", "date"],
        how="left",
    )

    home_features = features[features["is_home"] == True].copy()

    todays = _load_schedule(pred_date)
    if todays.empty:
        return pd.DataFrame()

    merged = home_features.merge(
        todays[["game_id", "home_team", "away_team", "date"]],
        on=["game_id", "date"],
        how="inner",
    )

    return merged


# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------


def _predict_spread(
    model, df_features: pd.DataFrame, feature_cols: list[str]
) -> pd.DataFrame:
    missing = [c for c in feature_cols if c not in df_features.columns]
    if missing:
        raise KeyError(f"[Spread] Missing feature columns: {missing}")

    X = df_features[feature_cols]
    preds = model.predict(X)

    out = df_features[["game_id", "date", "home_team", "away_team"]].copy()
    out["predicted_margin"] = preds
    return out


def _save_predictions(df: pd.DataFrame, pred_date: date, meta: dict):
    out_path = SPREAD_PRED_DIR / f"spread_{pred_date}.parquet"

    df = df.copy()
    df["model_name"] = meta["model_name"]
    df["model_version"] = meta["version"]
    df["feature_version"] = meta["feature_version"]

    df.to_parquet(out_path, index=False)
    logger.success(f"[Spread] Predictions saved → {out_path}")


# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------


def run_spread_prediction_for_date(pred_date: date | None = None) -> pd.DataFrame:
    pred_date = pred_date or date.today()
    logger.info(f"🏀 Running spread prediction for {pred_date}")

    model, meta = load_model_and_metadata("spread")
    feature_version = meta["feature_version"]
    feature_cols = meta["feature_cols"]

    df_features = _build_prediction_features(pred_date, feature_version)
    if df_features.empty:
        logger.warning(f"[Spread] No features available for {pred_date}.")
        return pd.DataFrame()

    pred_df = _predict_spread(model, df_features, feature_cols)
    _save_predictions(pred_df, pred_date, meta)

    logger.success(f"🏀 Spread prediction pipeline complete for {pred_date}")
    return pred_df


# Backwards compatibility alias
run_prediction_for_date = run_spread_prediction_for_date


if __name__ == "__main__":
    run_spread_prediction_for_date()
