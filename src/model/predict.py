"""
Prediction Pipeline
-------------------
Loads the latest trained model, builds features for a given date's games,
generates predictions, and saves them.

Fixes included:
- Remove broken `version` sorting and `is_production` assumptions
- Sort models by timestamp, fall back to latest model
- Build prediction features in long format (so `is_home` exists)
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import date

import pandas as pd
from loguru import logger

from src.config.paths import (
    MODEL_REGISTRY_DIR,
    SCHEDULE_SNAPSHOT,
    LONG_SNAPSHOT,
    ODDS_DIR,
    PREDICTIONS_DIR,
)
from src.features.builder import FeatureBuilder


# ---------------------------------------------------------
# Model registry helpers
# ---------------------------------------------------------


def _get_production_model_metadata() -> dict:
    registry_path = MODEL_REGISTRY_DIR / "index.json"

    if not registry_path.exists():
        raise FileNotFoundError("Model registry not found.")

    registry = json.loads(registry_path.read_text())
    models = registry.get("models", [])

    if not models:
        raise RuntimeError("No models found in registry.")

    # Prefer explicitly marked production models, if any
    prod_models = [m for m in models if m.get("is_production")]

    if prod_models:
        prod_models_sorted = sorted(prod_models, key=lambda m: m["timestamp"])
        return prod_models_sorted[-1]

    # Otherwise fall back to latest by timestamp
    logger.warning("No model marked as production. Falling back to latest version.")
    models_sorted = sorted(models, key=lambda m: m["timestamp"])
    return models_sorted[-1]


def _load_production_model():
    metadata = _get_production_model_metadata()
    model_path = Path(metadata["path"])

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = pd.read_pickle(model_path)
    return model, metadata


# ---------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------


def _load_todays_games(pred_date: date) -> pd.DataFrame:
    if not SCHEDULE_SNAPSHOT.exists():
        raise FileNotFoundError(f"Schedule snapshot not found: {SCHEDULE_SNAPSHOT}")

    df = pd.read_parquet(SCHEDULE_SNAPSHOT)
    df["date"] = pd.to_datetime(df["date"]).dt.date

    todays_games = df[df["date"] == pred_date]

    if todays_games.empty:
        logger.warning(f"No games found for {pred_date}.")
        return pd.DataFrame()

    return todays_games


def _load_todays_odds(pred_date: date) -> pd.DataFrame:
    odds_path = ODDS_DIR / f"odds_{pred_date}.parquet"

    if not odds_path.exists():
        logger.warning(f"No odds file found for {pred_date}: {odds_path}")
        return pd.DataFrame()

    return pd.read_parquet(odds_path)


# ---------------------------------------------------------
# Feature building for prediction
# ---------------------------------------------------------


def _expand_schedule_to_long(df_sched: pd.DataFrame) -> pd.DataFrame:
    """
    Convert schedule rows into long-format rows (team-centric),
    so we have `team`, `opponent`, and `is_home` like in training.
    """
    home_rows = df_sched.assign(
        team=df_sched["home_team"],
        opponent=df_sched["away_team"],
        is_home=True,
        points_for=df_sched.get("home_score", 0),
        points_against=df_sched.get("away_score", 0),
    )

    away_rows = df_sched.assign(
        team=df_sched["away_team"],
        opponent=df_sched["home_team"],
        is_home=False,
        points_for=df_sched.get("away_score", 0),
        points_against=df_sched.get("home_score", 0),
    )

    return pd.concat([home_rows, away_rows], ignore_index=True)


def _build_prediction_features(pred_date: date) -> pd.DataFrame:
    if not LONG_SNAPSHOT.exists():
        raise FileNotFoundError(f"Long-format snapshot not found: {LONG_SNAPSHOT}")

    # Historical long-format data used for feature engineering context
    df_long_hist = pd.read_parquet(LONG_SNAPSHOT)

    todays_games = _load_todays_games(pred_date)
    if todays_games.empty:
        return pd.DataFrame()

    # Convert today's schedule to team-centric long format
    todays_long = _expand_schedule_to_long(todays_games)

    # Build historical features
    fb = FeatureBuilder()
    hist_features = fb.build(df_long_hist)

    # Merge today's long-format rows with historical features.
    # We join on team/opponent so we can use lag/rolling stats from history.
    merged = todays_long.merge(
        hist_features,
        on=["team", "opponent"],
        how="left",
        suffixes=("_sched", ""),
    )

    return merged


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------


def _predict(model, features_df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    feature_cols = metadata["feature_cols"]

    missing = [c for c in feature_cols if c not in features_df.columns]
    if missing:
        raise KeyError(f"Missing required feature columns for prediction: {missing}")

    X = features_df[feature_cols]
    preds = model.predict_proba(X)[:, 1]

    features_df = features_df.copy()
    features_df["win_probability"] = preds
    return features_df


# ---------------------------------------------------------
# Saving
# ---------------------------------------------------------


def _save_predictions(pred_df: pd.DataFrame, pred_date: date):
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    out_path = PREDICTIONS_DIR / f"predictions_{pred_date}.parquet"
    pred_df.to_parquet(out_path, index=False)

    logger.success(f"Predictions saved → {out_path}")


# ---------------------------------------------------------
# Main entry point
# ---------------------------------------------------------


def run_prediction_for_date(pred_date: date | None = None):
    pred_date = pred_date or date.today()

    logger.info(f"=== Prediction Start for date={pred_date} ===")

    model, metadata = _load_production_model()

    features_df = _build_prediction_features(pred_date)
    if features_df.empty:
        logger.warning("No features available for prediction.")
        return None

    pred_df = _predict(model, features_df, metadata)
    _save_predictions(pred_df, pred_date)

    logger.success("=== Prediction Complete ===")
    return pred_df


if __name__ == "__main__":
    run_prediction_for_date()
