# ============================================================
# 🏀 NBA Analytics v3
# Module: Prediction Pipeline
# File: src/model/predict.py
# Author: Sadiq
#
# Description:
#     Loads the latest trained model from the registry, uses the
#     feature builder to construct the exact feature set used at
#     training time, and generates win probabilities for a given
#     target date's games. Ensures:
#       - schema alignment with training
#       - feature version consistency
#       - metadata-aware model selection
# ============================================================

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Tuple

import pandas as pd
from loguru import logger

from src.config.paths import (
    LONG_SNAPSHOT,
    MODEL_REGISTRY_DIR,
    SCHEDULE_SNAPSHOT,
    PREDICTIONS_DIR,
)
from src.features.builder import FeatureBuilder, FeatureConfig


def _get_latest_model_metadata() -> dict:
    index_path = MODEL_REGISTRY_DIR / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Model registry index not found at {index_path}")

    registry = json.loads(index_path.read_text())
    models = registry.get("models", [])
    if not models:
        raise RuntimeError("No models found in registry")

    # Prefer models marked as production; otherwise, pick latest by timestamp
    prod_models = [m for m in models if m.get("is_production")]
    if prod_models:
        models_to_consider = prod_models
    else:
        models_to_consider = models

    models_sorted = sorted(models_to_consider, key=lambda m: m["created_at_utc"])
    latest = models_sorted[-1]
    logger.info(
        f"Using model '{latest['model_name']}' version {latest['version']} "
        f"(production={latest.get('is_production', False)})"
    )
    return latest


def _load_model_and_metadata() -> Tuple[object, dict]:
    meta = _get_latest_model_metadata()
    model_path = Path(meta["path"])
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model = pd.read_pickle(model_path)
    return model, meta


def _load_schedule_for_date(pred_date: date) -> pd.DataFrame:
    if not SCHEDULE_SNAPSHOT.exists():
        raise FileNotFoundError(f"Schedule snapshot not found: {SCHEDULE_SNAPSHOT}")

    df = pd.read_parquet(SCHEDULE_SNAPSHOT)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    todays_games = df[df["date"] == pred_date].copy()

    if todays_games.empty:
        logger.warning(f"No games scheduled for {pred_date}.")
        return pd.DataFrame()

    return todays_games


def _expand_schedule_to_long(df_sched: pd.DataFrame) -> pd.DataFrame:
    home_rows = df_sched.assign(
        team=df_sched["home_team"],
        opponent=df_sched["away_team"],
        is_home=True,
    )
    away_rows = df_sched.assign(
        team=df_sched["away_team"],
        opponent=df_sched["home_team"],
        is_home=False,
    )

    long_like = pd.concat([home_rows, away_rows], ignore_index=True)
    return long_like[["game_id", "date", "team", "opponent", "is_home"]]


def _build_prediction_features(pred_date: date, feature_version: str) -> pd.DataFrame:
    """
    Builds prediction features for a specific date using the same
    feature builder and version as training. For historical dates,
    this aligns exactly with training. For future dates, you may
    need an incremental pre-game builder (future extension).
    """
    if not LONG_SNAPSHOT.exists():
        raise FileNotFoundError(f"Long-format snapshot not found: {LONG_SNAPSHOT}")

    df_long = pd.read_parquet(LONG_SNAPSHOT)
    df_long["date"] = pd.to_datetime(df_long["date"]).dt.date

    # For safety, restrict to games up to the prediction date
    df_hist = df_long[df_long["date"] <= pred_date].copy()
    if df_hist.empty:
        logger.warning(f"No historical long-format data available up to {pred_date}.")
        return pd.DataFrame()

    fb = FeatureBuilder(config=FeatureConfig(version=feature_version))
    features = fb.build_from_long(df_long=df_hist)

    # For a historical prediction date, features will include rows for that date.
    # We now filter to the game_ids/teams for the schedule of pred_date.
    todays_sched = _load_schedule_for_date(pred_date)
    if todays_sched.empty:
        return pd.DataFrame()

    todays_long_like = _expand_schedule_to_long(todays_sched)

    merged = todays_long_like.merge(
        features,
        on=["game_id", "team", "opponent", "date", "is_home"],
        how="left",
    )

    missing_features = merged[merged.isna().any(axis=1)]
    if not missing_features.empty:
        logger.warning(
            "Some prediction rows are missing features. "
            "These will be dropped. Examples:\n"
            f"{missing_features.head().to_string(index=False)}"
        )
        merged = merged.dropna()

    return merged


def _predict(model, features_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    missing_cols = [c for c in feature_cols if c not in features_df.columns]
    if missing_cols:
        raise KeyError(
            f"Missing feature columns in prediction dataframe: {missing_cols}"
        )

    X = features_df[feature_cols]
    if not hasattr(model, "predict_proba"):
        raise AttributeError("Loaded model does not support predict_proba().")

    proba = model.predict_proba(X)[:, 1]
    out = features_df.copy()
    out["win_probability"] = proba
    return out


def _save_predictions(pred_df: pd.DataFrame, pred_date: date, model_meta: dict):
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PREDICTIONS_DIR / f"predictions_{pred_date}.parquet"

    pred_df = pred_df.copy()
    pred_df["model_name"] = model_meta["model_name"]
    pred_df["model_version"] = model_meta["version"]
    pred_df["feature_version"] = model_meta["feature_version"]

    pred_df.to_parquet(out_path, index=False)
    logger.success(f"Predictions saved → {out_path}")


def run_prediction_for_date(pred_date: date | None = None):
    pred_date = pred_date or date.today()
    logger.info(f"🏀 Generating predictions for: {pred_date}")

    model, meta = _load_model_and_metadata()
    feature_version = meta.get("feature_version", "v1")
    feature_cols = meta["feature_cols"]

    features_df = _build_prediction_features(pred_date, feature_version=feature_version)
    if features_df.empty:
        logger.warning(f"No features available for prediction on {pred_date}.")
        return None

    pred_df = _predict(model, features_df, feature_cols)
    _save_predictions(pred_df, pred_date, meta)

    logger.success(f"🏀 Prediction pipeline complete for {pred_date}")
    return pred_df


if __name__ == "__main__":
    run_prediction_for_date()
