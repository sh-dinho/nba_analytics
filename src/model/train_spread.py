# ============================================================
# 🏀 NBA Analytics v3
# Module: Spread Model Training (ATS)
# File: src/model/train_spread.py
# Author: Sadiq
#
# Description:
#     Trains a regression model to predict margin of victory
#     (home_score - away_score). Only home rows are used so
#     each game appears exactly once.
#
#     Persists the model in the registry with metadata:
#       - model_type: "spread"
#       - target: "margin"
#       - market: "ats"
# ============================================================

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional
import json

import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from src.config.paths import LONG_SNAPSHOT, MODEL_REGISTRY_DIR
from src.features.builder import FeatureBuilder, FeatureConfig


MODEL_NAME_SPREAD = "rf_spread"


@dataclass
class SpreadTrainingConfig:
    feature_version: str = "v1"
    test_size: float = 0.2
    random_state: int = 42
    n_estimators: int = 300
    max_depth: Optional[int] = None
    min_samples_leaf: int = 2
    model_name: str = MODEL_NAME_SPREAD


def _load_long():
    df = pd.read_parquet(LONG_SNAPSHOT)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def _build_training_frame(cfg: SpreadTrainingConfig):
    df_long = _load_long()

    fb = FeatureBuilder(config=FeatureConfig(version=cfg.feature_version))
    features = fb.build_from_long(df_long=df_long)

    df = features.merge(
        df_long[["game_id", "team", "date", "points_for", "points_against", "is_home"]],
        on=["game_id", "team", "date"],
        how="left",
    )

    df_home = df[df["is_home"] == True].copy()  # noqa: E712
    df_home["margin"] = df_home["points_for"] - df_home["points_against"]

    df_home = df_home.dropna(subset=["margin"])
    return df_home


def _time_split(df, cfg):
    df = df.sort_values("date")
    unique_dates = sorted(df["date"].unique())
    cutoff = int(len(unique_dates) * (1 - cfg.test_size))
    cutoff_date = unique_dates[cutoff]

    train_df = df[df["date"] < cutoff_date]
    test_df = df[df["date"] >= cutoff_date]

    feature_cols = [
        c
        for c in df.columns
        if c not in ("margin", "date", "game_id", "team", "opponent", "is_home")
    ]

    X_train = train_df[feature_cols]
    y_train = train_df["margin"]
    X_test = test_df[feature_cols]
    y_test = test_df["margin"]

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        feature_cols,
        df["date"].min(),
        df["date"].max(),
    )


def _save_model(
    model,
    cfg,
    feature_cols,
    start_date,
    end_date,
    train_mae,
    test_mae,
    train_rmse,
    test_rmse,
):
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    model_dir = MODEL_REGISTRY_DIR / cfg.model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"{cfg.model_name}_{timestamp}.pkl"
    metadata_path = model_dir / f"{cfg.model_name}_{timestamp}.json"
    index_path = MODEL_REGISTRY_DIR / "index.json"

    pd.to_pickle(model, model_path)

    metadata = {
        "model_name": cfg.model_name,
        "model_type": "spread",
        "target": "margin",
        "market": "ats",
        "version": timestamp,
        "path": str(model_path),
        "created_at_utc": datetime.utcnow().isoformat(),
        "feature_version": cfg.feature_version,
        "feature_cols": feature_cols,
        "train_start_date": start_date.isoformat(),
        "train_end_date": end_date.isoformat(),
        "train_mae": train_mae,
        "test_mae": test_mae,
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "is_production": False,
    }

    metadata_path.write_text(json.dumps(metadata, indent=2))
    logger.success(f"[Spread] Model saved → {model_path}")

    if index_path.exists():
        registry = json.loads(index_path.read_text())
    else:
        registry = {"models": []}

    registry["models"].append(metadata)
    index_path.write_text(json.dumps(registry, indent=2))


def train_spread_model(cfg: Optional[SpreadTrainingConfig] = None):
    cfg = cfg or SpreadTrainingConfig()
    logger.info(f"🏀 Training spread model with config: {asdict(cfg)}")

    df = _build_training_frame(cfg)
    X_train, X_test, y_train, y_test, feature_cols, start_date, end_date = _time_split(
        df, cfg
    )

    model = RandomForestRegressor(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        min_samples_leaf=cfg.min_samples_leaf,
        random_state=cfg.random_state,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
    test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)

    _save_model(
        model,
        cfg,
        feature_cols,
        start_date,
        end_date,
        train_mae,
        test_mae,
        train_rmse,
        test_rmse,
    )

    logger.success("🏀 Spread model training complete.")
    return model


if __name__ == "__main__":
    train_spread_model()
