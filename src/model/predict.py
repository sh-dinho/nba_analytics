from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v4
# Module: Moneyline Prediction Pipeline
# File: src/model/predict.py
# Author: Sadiq
#
# Description:
#     Loads the latest moneyline model from the registry,
#     builds team-game features for a target date using the
#     canonical long snapshot and FeatureBuilder v4, and
#     generates win_probability for each team row.
#
#     v4 alignment:
#       - Uses LONG_SNAPSHOT (team-game rows)
#       - Uses FeatureBuilder(version=...)
#       - Uses Model Registry v4 (load_model_and_metadata)
#       - Two rows per game (home + away)
# ============================================================

from datetime import date
from typing import Any

import pandas as pd
from loguru import logger

from src.config.paths import LONG_SNAPSHOT, MONEYLINE_PRED_DIR
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


def _load_rows_for_date(pred_date: date) -> pd.DataFrame:
    """
    Load team-game rows for games on pred_date (both home and away).
    """
    df = _load_canonical_long()
    todays = df[df["date"].dt.date == pred_date].copy()

    if todays.empty:
        logger.warning(f"[Moneyline] No games scheduled for {pred_date}.")
    return todays


# ------------------------------------------------------------
# Feature building
# ------------------------------------------------------------


def _build_prediction_features(pred_date: date, feature_version: str) -> pd.DataFrame:
    """
    Build features for both home and away team rows on pred_date
    using only historical data up to (and including) pred_date.
    """
    df = _load_canonical_long()

    # Use history up to prediction date
    df_hist = df[df["date"].dt.date <= pred_date].copy()
    if df_hist.empty:
        logger.warning(f"[Moneyline] No historical data available up to {pred_date}.")
        return pd.DataFrame()

    fb = FeatureBuilder(version=feature_version)
    all_features = fb.build_from_long(df_hist)

    # Team rows for today's games
    todays = _load_rows_for_date(pred_date)
    if todays.empty:
        return pd.DataFrame()

    key_cols = ["game_id", "team", "date"]
    merged = todays[key_cols + ["opponent", "is_home"]].merge(
        all_features,
        on=key_cols,
        how="left",
    )

    missing_count = merged.isna().any(axis=1).sum()
    if missing_count:
        logger.debug(f"[Moneyline] Missing feature values for {missing_count} rows.")

    return merged


# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------


def _predict_moneyline(
    model: Any,
    df_features: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Run model.predict_proba() and return win_probability for each team row.
    """
    missing = [c for c in feature_cols if c not in df_features.columns]
    if missing:
        raise KeyError(f"[Moneyline] Missing feature columns: {missing}")

    X = df_features[feature_cols]
    probs = model.predict_proba(X)[:, 1]

    out = df_features[["game_id", "date", "team", "opponent", "is_home"]].copy()
    out["win_probability"] = probs
    return out


def _save_predictions(df: pd.DataFrame, pred_date: date, meta: dict) -> None:
    MONEYLINE_PRED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MONEYLINE_PRED_DIR / f"moneyline_{pred_date}.parquet"

    df = df.copy()
    df["model_type"] = meta["model_type"]
    df["model_version"] = meta["version"]
    df["feature_version"] = meta["feature_version"]

    df.to_parquet(out_path, index=False)
    logger.success(f"[Moneyline] Predictions saved → {out_path}")


# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------


def run_prediction_for_date(pred_date: date | None = None) -> pd.DataFrame:
    """
    Main entry point for moneyline predictions.

    Steps:
      1. Load latest moneyline model from registry
      2. Build team-game features for pred_date
      3. Predict win_probability for each team row
      4. Save predictions with model metadata
    """
    pred_date = pred_date or date.today()
    logger.info(f"🏀 Running moneyline prediction for {pred_date}")

    model, meta = load_model_and_metadata(model_type="moneyline")
    feature_version = meta["feature_version"]
    feature_cols = meta["feature_cols"]

    df_features = _build_prediction_features(pred_date, feature_version)
    if df_features.empty:
        logger.warning(f"[Moneyline] No features available for {pred_date}.")
        return pd.DataFrame()

    pred_df = _predict_moneyline(model, df_features, feature_cols)
    _save_predictions(pred_df, pred_date, meta)

    logger.success(f"🏀 Moneyline prediction pipeline complete for {pred_date}")
    return pred_df


if __name__ == "__main__":
    run_prediction_for_date()
