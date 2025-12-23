from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v4
# Module: Spread Prediction Pipeline (Margin Regression)
# File: src/model/predict_spread.py
# Author: Sadiq
#
# Description:
#     Loads the latest spread regression model from the registry,
#     builds home-team features for a target date using the
#     canonical long snapshot and FeatureBuilder v4, and
#     generates predicted_margin for each game
#     (home_team margin vs away_team).
#
#     v4 alignment:
#       - Uses LONG_SNAPSHOT (team-game rows)
#       - Uses FeatureBuilder(version=...)
#       - Uses Model Registry v4 (load_model_and_metadata)
#       - One row per game (home team only)
# ============================================================

from datetime import date
from typing import Any

import pandas as pd
from loguru import logger

from src.config.paths import LONG_SNAPSHOT, SPREAD_PRED_DIR
from src.features.builder import FeatureBuilder
from src.model.registry import load_model_and_metadata


# ------------------------------------------------------------
# Data loading
# ------------------------------------------------------------


def _load_canonical_long() -> pd.DataFrame:
    if not LONG_SNAPSHOT.exists():
        raise FileNotFoundError(f"Canonical long snapshot not found at {LONG_SNAPSHOT}")
    df = pd.read_parquet(LONG_SNAPSHOT)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _load_home_rows_for_date(pred_date: date) -> pd.DataFrame:
    """
    Load team-game rows for games on pred_date,
    restricted to home teams (is_home == 1), so each game
    appears exactly once.
    """
    df = _load_canonical_long()
    todays = df[(df["date"].dt.date == pred_date) & (df["is_home"] == 1)].copy()

    if todays.empty:
        logger.warning(f"[Spread] No games scheduled for {pred_date}.")
    return todays


# ------------------------------------------------------------
# Feature building
# ------------------------------------------------------------


def _build_prediction_features(pred_date: date, feature_version: str) -> pd.DataFrame:
    """
    Build features for home-team rows on pred_date using
    only historical data up to (and including) pred_date.
    """
    df = _load_canonical_long()

    # Use history up to prediction date
    df_hist = df[df["date"].dt.date <= pred_date].copy()
    if df_hist.empty:
        logger.warning(f"[Spread] No historical data available up to {pred_date}.")
        return pd.DataFrame()

    fb = FeatureBuilder(version=feature_version)
    all_features = fb.build_from_long(df_hist)

    # Home team rows for today's games
    todays_home = _load_home_rows_for_date(pred_date)
    if todays_home.empty:
        return pd.DataFrame()

    key_cols = ["game_id", "team", "date"]
    merged = todays_home[key_cols + ["opponent"]].merge(
        all_features,
        on=key_cols,
        how="left",
    )

    missing_count = merged.isna().any(axis=1).sum()
    if missing_count:
        logger.debug(f"[Spread] Missing feature values for {missing_count} home rows.")

    return merged


# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------


def _predict_spread(
    model: Any,
    df_features: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Run model.predict() and return predicted_margin for each game,
    where margin is home_score - away_score (from the model's training).
    """
    missing = [c for c in feature_cols if c not in df_features.columns]
    if missing:
        raise KeyError(f"[Spread] Missing feature columns: {missing}")

    X = df_features[feature_cols]
    preds = model.predict(X)

    out = df_features[["game_id", "date", "team", "opponent"]].copy()
    out = out.rename(columns={"team": "home_team", "opponent": "away_team"})
    out["predicted_margin"] = preds
    return out


def _save_predictions(df: pd.DataFrame, pred_date: date, meta: dict) -> None:
    SPREAD_PRED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SPREAD_PRED_DIR / f"spread_{pred_date}.parquet"

    df = df.copy()
    df["model_type"] = meta["model_type"]
    df["model_version"] = meta["version"]
    df["feature_version"] = meta["feature_version"]

    df.to_parquet(out_path, index=False)
    logger.success(f"[Spread] Predictions saved → {out_path}")


# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------


def run_spread_prediction_for_date(pred_date: date | None = None) -> pd.DataFrame:
    """
    Main entry point for spread regression predictions.

    Steps:
      1. Load latest spread regression model from registry
      2. Build home-team features for pred_date
      3. Predict margin (home - away) for each game
      4. Save predictions with model metadata
    """
    pred_date = pred_date or date.today()
    logger.info(f"🏀 Running spread prediction for {pred_date}")

    model, meta = load_model_and_metadata(model_type="spread_regression")
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
