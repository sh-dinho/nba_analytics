# ============================================================
# 🏀 NBA Analytics v3
# Module: Spread Prediction Pipeline
# File: src/model/predict_spread.py
# Author: Sadiq
#
# Description:
#     Predicts margin of victory for each game on a target date.
#     Produces:
#       - predicted_margin
#       - model metadata
# ============================================================

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
import pandas as pd
from loguru import logger

from src.config.paths import (
    LONG_SNAPSHOT,
    SCHEDULE_SNAPSHOT,
    MODEL_REGISTRY_DIR,
    DATA_DIR,
)
from src.features.builder import FeatureBuilder, FeatureConfig

SPREAD_PRED_DIR = DATA_DIR / "predictions_spread"
SPREAD_PRED_DIR.mkdir(parents=True, exist_ok=True)


def _get_latest_spread_meta():
    index_path = MODEL_REGISTRY_DIR / "index.json"
    registry = json.loads(index_path.read_text())
    models = [m for m in registry["models"] if m["model_type"] == "spread"]
    models = sorted(models, key=lambda m: m["created_at_utc"])
    return models[-1]


def _load_model(meta):
    return pd.read_pickle(meta["path"])


def _load_schedule(pred_date):
    df = pd.read_parquet(SCHEDULE_SNAPSHOT)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df[df["date"] == pred_date]


def _build_features(pred_date, feature_version):
    df_long = pd.read_parquet(LONG_SNAPSHOT)
    df_long["date"] = pd.to_datetime(df_long["date"]).dt.date
    df_hist = df_long[df_long["date"] <= pred_date]

    fb = FeatureBuilder(config=FeatureConfig(version=feature_version))
    features = fb.build_from_long(df_hist)

    features = features.merge(
        df_long[["game_id", "team", "date", "is_home"]],
        on=["game_id", "team", "date"],
        how="left",
    )

    return features[features["is_home"] == True]


def run_spread_prediction_for_date(pred_date: date | None = None):
    pred_date = pred_date or date.today()
    logger.info(f"🏀 Running spread prediction for {pred_date}")

    meta = _get_latest_spread_meta()
    model = _load_model(meta)
    feature_version = meta["feature_version"]
    feature_cols = meta["feature_cols"]

    df_sched = _load_schedule(pred_date)
    df_feat = _build_features(pred_date, feature_version)

    df = df_feat.merge(
        df_sched[["game_id", "home_team", "away_team", "date"]],
        on=["game_id", "date"],
        how="inner",
    )

    preds = model.predict(df[feature_cols])

    out = df[["game_id", "date", "home_team", "away_team"]].copy()
    out["predicted_margin"] = preds
    out["model_name"] = meta["model_name"]
    out["model_version"] = meta["version"]
    out["feature_version"] = meta["feature_version"]

    out_path = SPREAD_PRED_DIR / f"spread_{pred_date}.parquet"
    out.to_parquet(out_path, index=False)

    logger.success(f"[Spread] Predictions saved → {out_path}")
    return out


if __name__ == "__main__":
    run_spread_prediction_for_date()
