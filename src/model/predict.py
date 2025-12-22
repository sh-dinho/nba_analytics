# ============================================================
# 🏀 NBA Analytics v3
# Module: Moneyline Prediction Pipeline
# File: src/model/predict.py
# Author: Sadiq
#
# Description:
#     Loads the latest moneyline model from the registry,
#     builds features for a target date, and generates
#     win_probability predictions for each team.
#
#     This cleaned-up version standardizes:
#       - model loading
#       - metadata handling
#       - feature building
#       - prediction output
#       - compatibility with older helper names
# ============================================================

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Tuple, Dict, Any

import pandas as pd
from loguru import logger

from src.config.paths import (
    LONG_SNAPSHOT,
    SCHEDULE_SNAPSHOT,
    MODEL_REGISTRY_DIR,
    PREDICTIONS_DIR,
)
from src.features.builder import FeatureBuilder, FeatureConfig


# ------------------------------------------------------------
# Registry helpers
# ------------------------------------------------------------


def _load_registry() -> dict:
    """Load the model registry index.json."""
    index_path = MODEL_REGISTRY_DIR / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Model registry index not found at {index_path}")
    return json.loads(index_path.read_text())


def _get_latest_model_metadata(model_type: str = "moneyline") -> dict:
    """Return metadata for the latest model of a given type."""
    registry = _load_registry()
    models = registry.get("models", [])

    filtered = [m for m in models if m.get("model_type") == model_type]
    if not filtered:
        raise RuntimeError(f"No models of type '{model_type}' found in registry.")

    # Prefer production models if available
    prod = [m for m in filtered if m.get("is_production")]
    candidates = prod or filtered

    # Sort by creation time
    latest = sorted(candidates, key=lambda m: m["created_at_utc"])[-1]

    logger.info(
        f"[Moneyline] Using model '{latest['model_name']}' version {latest['version']} "
        f"(production={latest.get('is_production', False)})"
    )
    return latest


def _load_model_and_metadata() -> Tuple[Any, dict]:
    """Load the latest moneyline model and its metadata."""
    meta = _get_latest_model_metadata("moneyline")
    model_path = Path(meta["path"])

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model = pd.read_pickle(model_path)
    return model, meta


# Backwards compatibility alias
_load_model_and_meta = _load_model_and_metadata


# ------------------------------------------------------------
# Data loading
# ------------------------------------------------------------


def _load_schedule(pred_date: date) -> pd.DataFrame:
    """Load schedule rows for the target date."""
    df = pd.read_parquet(SCHEDULE_SNAPSHOT)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    todays = df[df["date"] == pred_date].copy()

    if todays.empty:
        logger.warning(f"[Moneyline] No games scheduled for {pred_date}.")
    return todays


def _load_long() -> pd.DataFrame:
    """Load canonical long-format dataset."""
    df = pd.read_parquet(LONG_SNAPSHOT)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


# ------------------------------------------------------------
# Feature building
# ------------------------------------------------------------


def _build_prediction_features(pred_date: date, feature_version: str) -> pd.DataFrame:
    """
    Build team-level features for all teams playing on pred_date.
    """
    df_long = _load_long()
    df_hist = df_long[df_long["date"] <= pred_date].copy()

    if df_hist.empty:
        logger.warning(f"[Moneyline] No historical data available up to {pred_date}.")
        return pd.DataFrame()

    fb = FeatureBuilder(config=FeatureConfig(version=feature_version))
    features = fb.build_from_long(df_hist)

    # Join schedule to get only teams playing today
    todays = _load_schedule(pred_date)
    if todays.empty:
        return pd.DataFrame()

    # Map game_id -> opponent
    opp_map = todays[["game_id", "home_team", "away_team"]].assign(
        home_opponent=lambda df: df["away_team"],
        away_opponent=lambda df: df["home_team"],
    )

    # Expand to team rows
    team_rows = pd.DataFrame(
        {
            "game_id": todays["game_id"].tolist() * 2,
            "team": todays["home_team"].tolist() + todays["away_team"].tolist(),
            "opponent": todays["away_team"].tolist() + todays["home_team"].tolist(),
            "date": pred_date,
        }
    )

    merged = team_rows.merge(
        features,
        on=["game_id", "team", "date"],
        how="left",
    )

    missing = (
        merged["win_probability"].isna().sum() if "win_probability" in merged else None
    )
    if missing:
        logger.debug(f"[Moneyline] Missing features for {missing} rows.")

    return merged


# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------


def _predict_moneyline(
    model, df_features: pd.DataFrame, feature_cols: list[str]
) -> pd.DataFrame:
    """Run model.predict() and return win_probability."""
    missing = [c for c in feature_cols if c not in df_features.columns]
    if missing:
        raise KeyError(f"Missing feature columns for prediction: {missing}")

    X = df_features[feature_cols]
    preds = model.predict(X)

    out = df_features[["game_id", "team", "opponent", "date"]].copy()
    out["win_probability"] = preds
    return out


def _save_predictions(df: pd.DataFrame, pred_date: date, meta: dict):
    """Save predictions to parquet with metadata."""
    out_path = PREDICTIONS_DIR / f"predictions_{pred_date}.parquet"

    df = df.copy()
    df["model_name"] = meta["model_name"]
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
    """
    pred_date = pred_date or date.today()
    logger.info(f"🏀 Running moneyline prediction for {pred_date}")

    model, meta = _load_model_and_metadata()
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
