# ============================================================
# 🏀 NBA Analytics v3
# Module: Totals Prediction Pipeline
# File: src/model/predict_totals.py
# Author: Sadiq
#
# Description:
#     Loads the latest totals model (model_type="totals") from
#     the registry, builds home-team game-level features using
#     the same feature version as training, and generates
#     predicted total points for each game on a target date.
#
#     Outputs one row per game:
#       - game_id
#       - date
#       - home_team
#       - away_team
#       - predicted_total
#       - model_name, model_version, feature_version
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
    SCHEDULE_SNAPSHOT,
    MODEL_REGISTRY_DIR,
    DATA_DIR,
)
from src.features.builder import FeatureBuilder, FeatureConfig

TOTALS_PRED_DIR = DATA_DIR / "predictions_totals"
TOTALS_PRED_DIR.mkdir(parents=True, exist_ok=True)


def _get_latest_totals_metadata() -> dict:
    index_path = MODEL_REGISTRY_DIR / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Model registry index not found at {index_path}")

    registry = json.loads(index_path.read_text())
    models = registry.get("models", [])
    if not models:
        raise RuntimeError("No models in registry.")

    totals_models = [m for m in models if m.get("model_type") == "totals"]
    if not totals_models:
        raise RuntimeError("No totals models found in registry.")

    prod_models = [m for m in totals_models if m.get("is_production")]
    models_to_consider = prod_models or totals_models

    models_sorted = sorted(models_to_consider, key=lambda m: m["created_at_utc"])
    latest = models_sorted[-1]

    logger.info(
        f"[Totals] Using model '{latest['model_name']}' version {latest['version']} "
        f"(production={latest.get('is_production', False)})"
    )
    return latest


def _load_totals_model_and_meta() -> Tuple[object, dict]:
    meta = _get_latest_totals_metadata()
    model_path = Path(meta["path"])
    if not model_path.exists():
        raise FileNotFoundError(f"Totals model file not found at {model_path}")

    model = pd.read_pickle(model_path)
    return model, meta


def _load_schedule(pred_date: date) -> pd.DataFrame:
    if not SCHEDULE_SNAPSHOT.exists():
        raise FileNotFoundError(f"Schedule snapshot not found: {SCHEDULE_SNAPSHOT}")

    df = pd.read_parquet(SCHEDULE_SNAPSHOT)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    todays = df[df["date"] == pred_date].copy()

    if todays.empty:
        logger.warning(f"[Totals] No games scheduled for {pred_date}.")
        return pd.DataFrame()

    return todays


def _build_prediction_features(pred_date: date, feature_version: str) -> pd.DataFrame:
    """
    For totals prediction, we:
      - build features from long-format data up to pred_date
      - restrict to home rows (is_home==True)
      - join with schedule so we have home/away team labels
    """
    if not LONG_SNAPSHOT.exists():
        raise FileNotFoundError(f"Long-format snapshot not found: {LONG_SNAPSHOT}")

    df_long = pd.read_parquet(LONG_SNAPSHOT)
    df_long["date"] = pd.to_datetime(df_long["date"]).dt.date

    df_hist = df_long[df_long["date"] <= pred_date].copy()
    if df_hist.empty:
        logger.warning(f"[Totals] No historical long-format data up to {pred_date}.")
        return pd.DataFrame()

    fb = FeatureBuilder(config=FeatureConfig(version=feature_version))
    features = fb.build_from_long(df_long=df_hist)

    # Merge back is_home and opponent to identify home rows
    features = features.merge(
        df_long[["game_id", "team", "date", "is_home"]],
        on=["game_id", "team", "date"],
        how="left",
    )

    df_home = features[features["is_home"] == True].copy()  # noqa: E712

    todays_sched = _load_schedule(pred_date)
    if todays_sched.empty:
        return pd.DataFrame()

    # Map game_id -> home_team, away_team
    game_meta = todays_sched[
        ["game_id", "home_team", "away_team", "date"]
    ].drop_duplicates()

    merged = df_home.merge(
        game_meta,
        on=["game_id", "date"],
        how="inner",
    )

    if merged.empty:
        logger.warning(f"[Totals] No matching home rows for schedule on {pred_date}.")
        return pd.DataFrame()

    return merged


def _predict_totals(
    model, df_features: pd.DataFrame, feature_cols: list[str]
) -> pd.DataFrame:
    missing_cols = [c for c in feature_cols if c not in df_features.columns]
    if missing_cols:
        raise KeyError(
            f"[Totals] Missing feature columns in prediction dataframe: {missing_cols}"
        )

    X = df_features[feature_cols]
    preds = model.predict(X)

    out = df_features[["game_id", "date", "home_team", "away_team"]].copy()
    out["predicted_total"] = preds
    return out


def _save_totals_predictions(df: pd.DataFrame, pred_date: date, meta: dict):
    out_path = TOTALS_PRED_DIR / f"totals_{pred_date}.parquet"

    df = df.copy()
    df["model_name"] = meta["model_name"]
    df["model_version"] = meta["version"]
    df["feature_version"] = meta["feature_version"]

    df.to_parquet(out_path, index=False)
    logger.success(f"[Totals] Predictions saved → {out_path}")


def run_totals_prediction_for_date(pred_date: date | None = None):
    pred_date = pred_date or date.today()
    logger.info(f"🏀 Running totals prediction for {pred_date}")

    model, meta = _load_totals_model_and_meta()
    feature_version = meta.get("feature_version", "v1")
    feature_cols = meta["feature_cols"]

    df_features = _build_prediction_features(pred_date, feature_version)
    if df_features.empty:
        logger.warning(
            f"[Totals] No features available for totals prediction on {pred_date}."
        )
        return None

    pred_df = _predict_totals(model, df_features, feature_cols)
    _save_totals_predictions(pred_df, pred_date, meta)

    logger.success(f"🏀 Totals prediction pipeline complete for {pred_date}")
    return pred_df


if __name__ == "__main__":
    run_totals_prediction_for_date()
